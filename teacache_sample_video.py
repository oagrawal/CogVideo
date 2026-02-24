import argparse
import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Any, Dict, Optional, Tuple,  Union
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, scale_lora_layers, unscale_lora_layers, export_to_video, load_image, logging as diffusers_logging

logger = diffusers_logging.get_logger(__name__)
from diffusers import CogVideoXPipeline, CogVideoXImageToVideoPipeline

coefficients_dict = {
    "CogVideoX-2b":[-3.10658903e+01,  2.54732368e+01, -5.92380459e+00,  1.75769064e+00, -3.61568434e-03],
    "CogVideoX-5b":[-1.53880483e+03,  8.43202495e+02, -1.34363087e+02,  7.97131516e+00, -5.23162339e-02],
    "CogVideoX-5b-I2V":[-1.53880483e+03,  8.43202495e+02, -1.34363087e+02,  7.97131516e+00, -5.23162339e-02],
    "CogVideoX1.5-5B":[ 2.50210439e+02, -1.65061612e+02,  3.57804877e+01, -7.81551492e-01, 3.58559703e-02],
    "CogVideoX1.5-5B-I2V":[ 1.22842302e+02, -1.04088754e+02,  2.62981677e+01, -3.06009921e-01, 3.71213220e-02],
}


def teacache_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        ofs: Optional[Union[int, float, torch.LongTensor]] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        if self.ofs_embedding is not None:
            ofs_emb = self.ofs_proj(ofs)
            ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
            ofs_emb = self.ofs_embedding(ofs_emb)
            emb = emb + ofs_emb

        # 2. Patch embedding
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        if self.enable_teacache:
            no_cache_mode = self.rel_l1_thresh == 0
            adaptive_switch = getattr(self, "adaptive_rel_l1_switch_step", None)
            if adaptive_switch is not None:
                if self.cnt < adaptive_switch:
                    current_thresh = getattr(self, "adaptive_rel_l1_thresh_1", self.rel_l1_thresh)
                else:
                    current_thresh = getattr(self, "adaptive_rel_l1_thresh_2", self.rel_l1_thresh)
            else:
                current_thresh = self.rel_l1_thresh
            delta_temni_list = getattr(self, "delta_TEMNI", None)
            rescale_func = np.poly1d(self.coefficients)
            if delta_temni_list is not None and isinstance(delta_temni_list, list):
                if self.previous_modulated_input is not None:
                    raw_delta = ((emb - self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item()
                    delta_temni_list.append(float(rescale_func(raw_delta)))
                else:
                    delta_temni_list.append(0.0)
            if self.cnt == 0 or self.cnt == self.num_steps-1:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
            elif no_cache_mode:
                should_calc = True
            else:
                self.accumulated_rel_l1_distance += rescale_func(((emb-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
                if self.accumulated_rel_l1_distance < current_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    self.accumulated_rel_l1_distance = 0
            self.previous_modulated_input = emb
            self.cnt += 1
            if self.cnt == self.num_steps:
                self.cnt = 0            
        
        if self.enable_teacache:
            if not should_calc:
                hidden_states += self.previous_residual
                encoder_hidden_states += self.previous_residual_encoder
            else:
                ori_hidden_states = hidden_states.clone()
                ori_encoder_hidden_states = encoder_hidden_states.clone()
                # 4. Transformer blocks
                for i, block in enumerate(self.transformer_blocks):
                    if torch.is_grad_enabled() and self.gradient_checkpointing:

                        def create_custom_forward(module):
                            def custom_forward(*inputs):
                                return module(*inputs)

                            return custom_forward

                        ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                        hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            hidden_states,
                            encoder_hidden_states,
                            emb,
                            image_rotary_emb,
                            **ckpt_kwargs,
                        )
                    else:
                        hidden_states, encoder_hidden_states = block(
                            hidden_states=hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                            temb=emb,
                            image_rotary_emb=image_rotary_emb,
                        )

                self.previous_residual = hidden_states - ori_hidden_states
                self.previous_residual_encoder = encoder_hidden_states - ori_encoder_hidden_states
        else:
            # 4. Transformer blocks
            for i, block in enumerate(self.transformer_blocks):
                if torch.is_grad_enabled() and self.gradient_checkpointing:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        emb,
                        image_rotary_emb,
                        **ckpt_kwargs,
                    )
                else:
                    hidden_states, encoder_hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=emb,
                        image_rotary_emb=image_rotary_emb,
                    )

        if not self.config.use_rotary_positional_embeddings:
            # CogVideoX-2B
            hidden_states = self.norm_final(hidden_states)
        else:
            # CogVideoX-5B and CogvideoX1.5-5B
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            hidden_states = self.norm_final(hidden_states)
            hidden_states = hidden_states[:, text_seq_length:]

        # 5. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 6. Unpatchify
        p = self.config.patch_size
        p_t = self.config.patch_size_t

        if p_t is None:
            output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
            output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
        else:
            output = hidden_states.reshape(
                batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
            )
            output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)


def run_generation(
    prompt,
    seed,
    rel_l1_thresh,
    save_file,
    ckpts_path="THUDM/CogVideoX1.5-5B",
    num_inference_steps=50,
    negative_prompt=None,
    height=768,
    width=1360,
    num_frames=81,
    guidance_scale=6.0,
    fps=16,
    pipe=None,
    skip_delta_plot=False,
    adaptive_schedule=None,
):
    """Generate a single video. Returns pipe for reuse in batch. Used by batch_generate_cogvideo.py."""
    mode = ckpts_path.split("/")[-1]
    if pipe is None:
        pipe = CogVideoXPipeline.from_pretrained(ckpts_path, torch_dtype=torch.bfloat16)
        pipe.to("cuda")
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
    cls = pipe.transformer.__class__
    cls.enable_teacache = True
    cls.rel_l1_thresh = rel_l1_thresh
    cls.accumulated_rel_l1_distance = 0
    cls.previous_modulated_input = None
    cls.previous_residual = None
    cls.previous_residual_encoder = None
    cls.num_steps = num_inference_steps
    cls.cnt = 0
    cls.coefficients = coefficients_dict[mode]
    cls.forward = teacache_forward
    # reset any previous adaptive schedule
    cls.adaptive_rel_l1_switch_step = None
    cls.adaptive_rel_l1_thresh_1 = None
    cls.adaptive_rel_l1_thresh_2 = None
    if adaptive_schedule is not None:
        cls.adaptive_rel_l1_switch_step = adaptive_schedule.get("switch_step")
        cls.adaptive_rel_l1_thresh_1 = adaptive_schedule.get("thresh1", rel_l1_thresh)
        cls.adaptive_rel_l1_thresh_2 = adaptive_schedule.get("thresh2", rel_l1_thresh)
    if rel_l1_thresh == 0:
        pipe.transformer.__class__.delta_TEMNI = []

    video = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        use_dynamic_cfg=True,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator("cuda").manual_seed(seed),
    ).frames[0]

    os.makedirs(os.path.dirname(os.path.abspath(save_file)), exist_ok=True)
    export_to_video(video, save_file, fps=fps)
    if rel_l1_thresh == 0 and not skip_delta_plot:
        _plot_delta_temni(pipe.transformer.__class__, save_file)
    return pipe


def main(args):
    seed = args.seed
    ckpts_path = args.ckpts_path
    output_path = args.output_path
    num_inference_steps = args.num_inference_steps
    rel_l1_thresh = args.rel_l1_thresh
    generate_type = args.generate_type
    prompt = args.prompt
    negative_prompt = args.negative_prompt
    height = args.height
    width = args.width
    num_frames = args.num_frames
    guidance_scale = args.guidance_scale
    fps = args.fps
    image_path = args.image_path
    mode = ckpts_path.split("/")[-1]
    os.makedirs(output_path, exist_ok=True)

    if generate_type == "t2v":
        pipe = CogVideoXPipeline.from_pretrained(ckpts_path, torch_dtype=torch.bfloat16)
    else:
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(ckpts_path, torch_dtype=torch.bfloat16)
        image = load_image(image=image_path)

    # TeaCache
    pipe.transformer.__class__.enable_teacache = True
    pipe.transformer.__class__.rel_l1_thresh = rel_l1_thresh
    pipe.transformer.__class__.accumulated_rel_l1_distance = 0
    pipe.transformer.__class__.previous_modulated_input = None
    pipe.transformer.__class__.previous_residual = None
    pipe.transformer.__class__.previous_residual_encoder = None
    pipe.transformer.__class__.num_steps = num_inference_steps
    pipe.transformer.__class__.cnt = 0
    pipe.transformer.__class__.coefficients = coefficients_dict[mode]
    pipe.transformer.__class__.forward = teacache_forward
    if rel_l1_thresh == 0:
        pipe.transformer.__class__.delta_TEMNI = []

    pipe.to("cuda")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    if generate_type == "t2v":
        video = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator("cuda").manual_seed(seed)
        ).frames[0]
    else:
        video = pipe(
            height=height,
            width=width,
            prompt=prompt,
            image=image,
            num_inference_steps=num_inference_steps,  # Number of inference steps
            num_frames=num_frames,  # Number of frames to generate
            use_dynamic_cfg=True,  # This id used for DPM scheduler, for DDIM scheduler, it should be False
            guidance_scale=guidance_scale,
            generator=torch.Generator("cuda").manual_seed(seed),  # Set the seed for reproducibility
        ).frames[0]
    words = prompt.split()[:5]
    video_path = f"{output_path}/teacache_{mode}_{'_'.join(words)}_{rel_l1_thresh}.mp4"
    export_to_video(video, video_path, fps=fps)
    if rel_l1_thresh == 0:
        _plot_delta_temni(pipe.transformer.__class__, video_path)


def _plot_delta_temni(transformer_class, save_file):
    """Plot and save delta TEMNI when recorded (no TeaCache). Same style as Wan/Mochi/HunyuanVideo."""
    delta_temni = getattr(transformer_class, "delta_TEMNI", None)
    if not delta_temni or len(delta_temni) == 0:
        return
    out_dir = os.path.dirname(os.path.abspath(save_file))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = "."
    base = os.path.splitext(os.path.basename(save_file))[0]
    plt.figure(figsize=(10, 6))
    x = range(1, len(delta_temni) + 1)
    plt.plot(x, delta_temni, "g-", linewidth=2, marker="s", markersize=4)
    plt.xlabel("Forward step (1 = first solver step, high t → right = last step, low t)")
    plt.ylabel("Delta TEMNI (rescaled relative L1)")
    plt.title("Delta TEMNI over diffusion steps (no TeaCache)")
    plt.grid(True, alpha=0.3)
    ax = plt.gca()
    step = max(1, len(delta_temni) // 20)
    ax.xaxis.set_major_locator(plt.MultipleLocator(step))
    plot_path = os.path.join(out_dir, f"{base}_delta_TEMNI_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Delta TEMNI plot saved to: {plot_path}")
    txt_path = os.path.join(out_dir, f"{base}_delta_TEMNI.txt")
    with open(txt_path, "w") as f:
        for v in delta_temni:
            f.write(f"{v}\n")
    print(f"Delta TEMNI values saved to: {txt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CogVideoX1.5-5B with given parameters")
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of inference steps')
    parser.add_argument("--output_path", type=str, default="./teacache_results", help="The path where the generated video will be saved")
    parser.add_argument('--ckpts_path', type=str, default="THUDM/CogVideoX1.5-5B", help='Path to checkpoint, t2v->THUDM/CogVideoX1.5-5B, i2v->THUDM/CogVideoX1.5-5B-I2V')
    parser.add_argument('--rel_l1_thresh', type=float, default=0.2, help='Higher speedup will cause to worse quality -- 0.1 for 1.3x speedup -- 0.2 for 1.8x speedup -- 0.3 for 2.1x speedup')
    parser.add_argument('--prompt', type=str, default="A clear, turquoise river flows through a rocky canyon, cascading over a small waterfall and forming a pool of water at the bottom.The river is the main focus of the scene, with its clear water reflecting the surrounding trees and rocks. The canyon walls are steep and rocky, with some vegetation growing on them. The trees are mostly pine trees, with their green needles contrasting with the brown and gray rocks. The overall tone of the scene is one of peace and tranquility.", help='Description of the video for the model to generate')
    parser.add_argument('--negative_prompt', type=str, default=None, help='Description of unwanted situations in model generated videos')
    parser.add_argument("--image_path",type=str,default=None,help="The path of the image to be used as the background of the video")
    parser.add_argument("--generate_type", type=str, default="t2v", help="The type of video generation (e.g., 't2v', 'i2v')")
    parser.add_argument("--width", type=int, default=1360, help="Number of steps for the inference process")
    parser.add_argument("--height", type=int, default=768, help="Number of steps for the inference process")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of steps for the inference process")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument("--fps", type=int, default=16, help="Frame rate of video")
    args = parser.parse_args()

    main(args)