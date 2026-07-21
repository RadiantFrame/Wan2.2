# -*- coding: utf-8 -*-
# @author: Jiang Wei
# @date: 2026/06/18
# @description: Generate VBench evaluation videos for all dimensions using Wan2.2 TI2V/T2V pipeline.
#   Follows the "Evaluate All Dimensions" specification in VBench/prompts/README.md.
#   The model is loaded ONCE per worker process and reused across all tasks
#   assigned to that worker (no per-sample subprocess spawn).
#**************************************************************************************
import os
import sys
import argparse
import logging
import multiprocessing
import tqdm

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def parse_args():
    """"""
    parser = argparse.ArgumentParser(
        description="Generate VBench evaluation videos for all dimensions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--prompt_path", type=str,
        default="/mnt/luci-nas/jiangwei/projects/VBench/prompts/all_dimension.txt",
        help="Path to the VBench all_dimension prompt list"
    )
    parser.add_argument(
        "--save_path", type=str, required=True,
        help="Directory to save generated videos"
    )
    parser.add_argument(
        "--log_dir", type=str, default="./logs",
        help="Directory to save log files"
    )
    parser.add_argument(
        "--num_samples", type=int, default=5,
        help="Number of videos to sample per prompt"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed for reproducibility"
    )
    parser.add_argument(
        "--task", type=str, default="ti2v-5B",
        help="Wan2.2 task type (only t2v-* / ti2v-* supported here)"
    )
    parser.add_argument(
        "--size", type=str, default="1280*704",
        help="Video resolution (width*height)"
    )
    parser.add_argument(
        "--ckpt_dir", type=str,
        default="/mnt/luci-nas/jiangwei/hf_hub/Wan2.2-TI2V-5B",
        help="Path to the Wan2.2 checkpoint directory"
    )
    # multiprocessing
    parser.add_argument(
        "--n_proc", type=int, default=1,
        help="Number of parallel worker processes"
    )
    parser.add_argument(
        "--gpu_ids", type=str, default="0,1,2,3",
        help="Comma-separated GPU ids, each worker is assigned one GPU (round-robin)"
    )
    # generation knobs (forwarded to the pipeline; None => use cfg default)
    parser.add_argument("--frame_num", type=int, default=None)
    parser.add_argument("--sample_steps", type=int, default=None)
    parser.add_argument("--sample_shift", type=float, default=None)
    parser.add_argument("--sample_guide_scale", type=float, default=None)
    parser.add_argument("--sample_solver", type=str, default="unipc",
                        choices=["unipc", "dpm++"])
    parser.add_argument("--offload_model", action="store_true", default=False)
    parser.add_argument("--convert_model_dtype", action="store_true", default=False)
    parser.add_argument("--t5_cpu", action="store_true", default=False)
    return parser.parse_args()


def _init_logging(log_file=None):
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=handlers,
    )


def process(proc_ordinal, queue, gpu_id, args, tasks):
    """
    Worker function executed in each subprocess.
    Loads the Wan pipeline ONCE, then generates all assigned videos in a loop.
    :param tasks: list of (prompt, index, seed, save_file) tuples assigned to this worker.
    """
    # Pin this worker to the assigned GPU BEFORE importing torch.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Heavy imports happen here so the parent process never touches CUDA
    # (safe with the default `fork` start method on Linux).
    import torch
    # `wan` package lives at the repo root (one level above this script's dir).
    _REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS
    from wan.text2video import WanT2V
    from wan.textimage2video import WanTI2V
    from wan.utils.utils import save_video

    _init_logging(log_file=os.path.join(args.log_dir, f"worker_{proc_ordinal}.log"))

    cfg = WAN_CONFIGS[args.task]
    # fill in cfg defaults for unspecified knobs
    frame_num = args.frame_num if args.frame_num is not None else cfg.frame_num
    sample_steps = args.sample_steps if args.sample_steps is not None else cfg.sample_steps
    sample_shift = args.sample_shift if args.sample_shift is not None else cfg.sample_shift
    sample_guide_scale = args.sample_guide_scale if args.sample_guide_scale is not None else cfg.sample_guide_scale

    logging.info(f"[worker {proc_ordinal} / GPU {gpu_id}] Building pipeline (task={args.task}) ...")

    # Build the pipeline once per worker. We are always single-GPU per worker,
    # so no FSDP / sequence-parallel / distributed init is required.
    common_kwargs = dict(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=0,  # CUDA_VISIBLE_DEVICES has been remapped; local index is 0
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
    )

    if "ti2v" in args.task:
        pipeline = WanTI2V(**common_kwargs)
        pipeline_kind = "ti2v"
    elif "t2v" in args.task:
        pipeline = WanT2V(**common_kwargs)
        pipeline_kind = "t2v"
    else:
        raise NotImplementedError(
            f"task={args.task} is not supported by this script (use t2v-* or ti2v-*)"
        )

    logging.info(f"[worker {proc_ordinal} / GPU {gpu_id}] Pipeline ready, processing {len(tasks)} tasks.")

    success_count = 0
    skip_count = 0
    fail_count = 0

    def _is_valid_video(path, expected_frames):
        """Check if a video file exists and has the expected number of frames."""
        if not os.path.isfile(path):
            return False
        if os.path.getsize(path) < 1024 * 100:  # < 100KB is likely corrupted
            logging.info(f"[GPU {gpu_id}] skip (empty): {os.path.basename(path)}")
            return False
        try:
            import av
            container = av.open(path)
            stream = container.streams.video[0]
            frame_count = stream.frames
            container.close()
            if frame_count != expected_frames:
                logging.info(
                    f"[GPU {gpu_id}] skip (frame mismatch): {os.path.basename(path)} "
                    f"expected={expected_frames} got={frame_count}"
                )
                return False
        except Exception as e:
            logging.info(f"[GPU {gpu_id}] error occurred while checking video {os.path.basename(path)}: {e}")
            return False
        return True

    # inference timing (exclude the first generated video as warmup)
    import time
    timed_total = 0.0   # seconds
    timed_count = 0     # number of videos counted
    is_first_generated = True

    for prompt, index, seed, save_file in tqdm.tqdm(
        tasks, desc=f"[GPU {gpu_id}] worker {proc_ordinal}"
    ):
        # skip if video already exists and is valid
        if _is_valid_video(save_file, expected_frames=frame_num):
            skip_count += 1
            logging.info(f"[GPU {gpu_id}] skip (exists): {os.path.basename(save_file)}")
            continue

        try:
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            if pipeline_kind == "ti2v":
                video = pipeline.generate(
                    prompt,
                    img=None,
                    size=SIZE_CONFIGS[args.size],
                    max_area=MAX_AREA_CONFIGS[args.size],
                    frame_num=frame_num,
                    shift=sample_shift,
                    sample_solver=args.sample_solver,
                    sampling_steps=sample_steps,
                    guide_scale=sample_guide_scale,
                    seed=seed,
                    offload_model=args.offload_model,
                )
            else:  # t2v
                video = pipeline.generate(
                    prompt,
                    size=SIZE_CONFIGS[args.size],
                    frame_num=frame_num,
                    shift=sample_shift,
                    sample_solver=args.sample_solver,
                    sampling_steps=sample_steps,
                    guide_scale=sample_guide_scale,
                    seed=seed,
                    offload_model=args.offload_model,
                )

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            save_video(
                tensor=video[None],
                save_file=save_file,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )
            del video
            torch.cuda.synchronize()
            success_count += 1

            # skip the first successfully-generated video (warmup)
            if is_first_generated:
                is_first_generated = False
                logging.info(
                    f"[GPU {gpu_id}] warmup video done in {elapsed:.2f}s "
                    f"(excluded from timing stats)"
                )
            else:
                timed_total += elapsed
                timed_count += 1
                logging.info(
                    f"[GPU {gpu_id}] inference {elapsed:.2f}s | "
                    f"avg {timed_total / timed_count:.2f}s "
                    f"over {timed_count} videos"
                )
        except Exception as e:
            logging.exception(
                f"[GPU {gpu_id}] FAILED: prompt='{prompt}' seed={seed}: {e}"
            )
            fail_count += 1

    if timed_count > 0:
        logging.info(
            f"[worker {proc_ordinal} / GPU {gpu_id}] inference summary: "
            f"avg {timed_total / timed_count:.2f}s/video over {timed_count} videos "
            f"(first video excluded as warmup)"
        )

    # report result back to main process
    queue.put(('end', success_count, skip_count, fail_count, timed_total, timed_count))


# main function
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def main(args):
    """"""
    os.makedirs(args.save_path, exist_ok=True)

    # read prompt list (Evaluate All Dimensions)
    with open(args.prompt_path, 'r') as f:
        prompt_list = f.readlines()
    prompt_list = [prompt.strip() for prompt in prompt_list]
    print(f"Loaded {len(prompt_list)} prompts from {args.prompt_path}")

    # build full task list: (prompt, index, seed, save_file)
    all_tasks = []
    for prompt in prompt_list:
        for index in range(args.num_samples):
            seed = args.seed + hash(prompt + str(index)) % (2**31)
            save_file = os.path.join(args.save_path, f"{prompt}-{index}.mp4")
            all_tasks.append((prompt, index, seed, save_file))

    total = len(all_tasks)
    print(f"Total tasks: {total} (prompts={len(prompt_list)} x samples={args.num_samples})")

    # multiprocessing
    gpu_ids = list(map(int, args.gpu_ids.split(",")))

    if args.n_proc <= 1:
        # single-process mode
        gpu_id = gpu_ids[0]
        queue = multiprocessing.Queue()
        process(0, queue, gpu_id, args, all_tasks)
        _, success_count, skip_count, fail_count, timed_total, timed_count = queue.get()
    else:
        # multi-process mode: round-robin assignment to GPUs
        p_list = []
        queue = multiprocessing.Queue()

        for i in range(args.n_proc):
            gpu_id = gpu_ids[i % len(gpu_ids)]
            split_tasks = all_tasks[i::args.n_proc]
            p = multiprocessing.Process(
                target=process,
                args=(i, queue, gpu_id, args, split_tasks)
            )
            p.start()
            print(f"Started worker {i} (pid={p.pid}, gpu={gpu_id}, tasks={len(split_tasks)})")
            p_list.append(p)

        # collect results from all workers
        success_count = 0
        skip_count = 0
        fail_count = 0
        timed_total = 0.0
        timed_count = 0
        for _ in range(args.n_proc):
            msg, s, sk, f, tt, tc = queue.get()
            success_count += s
            skip_count += sk
            fail_count += f
            timed_total += tt
            timed_count += tc

        # wait for all processes to finish
        for p in p_list:
            p.join()

    print(f"Done. success={success_count}, skipped={skip_count}, failed={fail_count}")
    if timed_count > 0:
        print(
            f"Inference timing across all workers: "
            f"avg {timed_total / timed_count:.2f}s/video over {timed_count} videos "
            f"(first video of each worker excluded as warmup)"
        )
    else:
        print("Inference timing: no videos counted (all skipped, failed, or only warmup).")


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == "__main__":
    # Make sure the parent process never imports torch before forking,
    # so that each child can freely set CUDA_VISIBLE_DEVICES.
    main(parse_args())
