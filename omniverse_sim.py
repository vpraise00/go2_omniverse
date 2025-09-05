"""Play a trained checkpoint (RSL-RL) if available. No ROS2. Safe fallbacks."""
from __future__ import annotations

import os
import re
import glob
import argparse
import importlib
import traceback
import torch
import gymnasium as gym

def _locate_apps_dir() -> str | None:
    """Find IsaacLab/apps; prefer env var, else sibling 'IsaacLab/apps'."""
    # 1) explicit env
    env_dir = os.environ.get("ISAACLAB_APPS_DIR")
    if env_dir and os.path.isdir(env_dir):
        return env_dir
    # 2) sibling folder: <repo_root>/IsaacLab/apps
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.abspath(os.path.join(here, "..", "IsaacLab", "apps"))
    if os.path.isdir(cand):
        return cand
    # 3) fallback: None
    return None

def _ensure_exp_path_env():
    """Ensure EXP_PATH is set for AppLauncher to resolve experience files."""
    apps_dir = _locate_apps_dir()
    if apps_dir:
        os.environ.setdefault("EXP_PATH", apps_dir)
        os.environ.setdefault("ISAACLAB_APPS_DIR", apps_dir)

def _ensure_lab_registry():
    """Import Isaac Lab task registries so gym.make(<task>) works in 2.2+.
    Must be called after SimulationApp().
    """
    import importlib
    # Ensure core package and task registry are loaded (these register Gymnasium envs)
    importlib.import_module("isaaclab")
    importlib.import_module("isaaclab_tasks")

def _create_minimal_stage():
    """Create an empty stage with a ground plane (smoke)."""
    try:
        from omni.usd import get_context  # type: ignore
        from omni.kit.commands import execute  # type: ignore
        from pxr import UsdGeom  # type: ignore

        ctx = get_context()
        ctx.new_stage()                     # creates a new stage (no return value used)
        stage = ctx.get_stage()             # get the Usd.Stage
        # Optional: define and set default prim
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())

        # Set Z-up using TfToken
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

        # Add a ground plane
        execute("CreateMeshPrimWithDefaultXform", prim_type="Plane", prim_path="/World/groundPlane")
        print("[INFO] Minimal stage created.")
    except Exception as e:
        print(f"[WARN] Minimal stage setup failed: {e}")
        traceback.print_exc()


def _find_checkpoint(experiment_name: str, base_dir: str = "logs/rsl_rl", load_run: str | None = None,
                     load_checkpoint: str | None = None) -> str | None:
    """Find a checkpoint path:
       - If load_run and load_checkpoint are given, use them.
       - Else pick the latest run dir and highest numbered model_*.pt.
    """
    exp_dir = os.path.abspath(os.path.join(base_dir, experiment_name))
    if not os.path.isdir(exp_dir):
        print(f"[WARN] Experiment dir not found: {exp_dir}")
        return None

    run_dir = None
    if load_run:
        cand = os.path.join(exp_dir, load_run)
        if os.path.isdir(cand):
            run_dir = cand
        else:
            print(f"[WARN] Run dir not found: {cand}")
    if run_dir is None:
        # pick latest by mtime
        runs = [os.path.join(exp_dir, d) for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))]
        if not runs:
            print(f"[WARN] No run directories in {exp_dir}")
            return None
        run_dir = max(runs, key=os.path.getmtime)

    if load_checkpoint:
        ckpt = os.path.join(run_dir, load_checkpoint)
        if os.path.isfile(ckpt):
            return ckpt
        else:
            print(f"[WARN] Checkpoint not found: {ckpt}")

    # auto-pick highest model_*.pt
    candidates = glob.glob(os.path.join(run_dir, "model_*.pt"))
    if not candidates:
        print(f"[WARN] No checkpoints in {run_dir}")
        return None

    def _num(p):
        m = re.search(r"model_(\d+)\.pt$", os.path.basename(p))
        return int(m.group(1)) if m else -1

    return max(candidates, key=_num)

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Play trained policy (RSL-RL).")
    # Add Isaac Lab AppLauncher args (headless, rendering_mode, experience, enable_cameras, etc.)
    # Import here to avoid preloading omni modules at import time.
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)

    parser.add_argument("--robot", type=str, default="go2", choices=["go2", "g1"], help="Robot type.")
    parser.add_argument("--robot_amount", type=int, default=1, help="Number of robots (envs).")
    parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Unitree-Go2-v0",
                        help="Gym task id (optional).")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Override experiment_name for checkpoint search.")
    parser.add_argument("--load_run", type=str, default=None, help="Run folder name under logs/rsl_rl/<exp>/")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Checkpoint filename, e.g. model_7850.pt")
    return parser.parse_args(argv)


def run_sim(argv=None):
    # Accept EULA to avoid blocking dialogs
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "1")
    os.environ.setdefault("OMNI_KIT_FAST_SHUTDOWN", "1")
    _ensure_exp_path_env()

    # Defer import to runtime to avoid pre-SimulationApp warnings
    from isaaclab.app import AppLauncher

    args = parse_args(argv)
    # Start SimulationApp with Isaac Lab AppLauncher (autonomous selection of experience file/rendering mode)
    print("[INFO] Launching Isaac Sim via Isaac Lab AppLauncher ...")
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Try full RL playback path (requires your env + rsl_rl). On failure, fallback to smoke.
    try:
        _ensure_lab_registry()
        from rsl_rl.runners import OnPolicyRunner  # type: ignore

        env = None

        # Prefer robot-specific creation to avoid mismatched dims
        if args.robot == "g1":
            gym_id = "Isaac-Velocity-Rough-Unitree-G1-v0"
            print(f"[INFO] Creating gym env: {gym_id} (num_envs={args.robot_amount})")
            try:
                sim_device = "cuda:0" if torch.cuda.is_available() else "cpu"
                try:
                    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
                    env_cfg = parse_env_cfg(gym_id, device=sim_device, num_envs=args.robot_amount)
                    env = gym.make(gym_id, cfg=env_cfg)
                except Exception as e_cfg:
                    print(f"[WARN] parse_env_cfg failed for G1 ({e_cfg}). Falling back to manual ctor.")
            except Exception as e:
                print(f"[WARN] Gym registry for G1 failed: {e}. Falling back to manual ctor.")

            if env is None:
                # Manual ctor with G1 cfg
                from isaaclab.envs import ManagerBasedRLEnv
                try:
                    from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import G1RoughEnvCfg
                except Exception:
                    raise RuntimeError("G1RoughEnvCfg import failed; ensure isaaclab_tasks is installed.")
                cfg = G1RoughEnvCfg()
                cfg.scene.num_envs = args.robot_amount
                print(f"[INFO] Creating manual G1 env (num_envs={args.robot_amount})")
                env = ManagerBasedRLEnv(cfg)
            # default experiment name for checkpoint search
            exp_name_default = "g1_rough"

        elif args.task.startswith("Isaac-"):
            print(f"[INFO] Creating gym env: {args.task} (num_envs={args.robot_amount})")
            try:
                # Prefer cfg-based construction
                sim_device = "cuda:0" if torch.cuda.is_available() else "cpu"
                try:
                    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
                    env_cfg = parse_env_cfg(args.task, device=sim_device, num_envs=args.robot_amount)
                    env = gym.make(args.task, cfg=env_cfg)
                except Exception as e_cfg:
                    print(f"[WARN] parse_env_cfg failed ({e_cfg}). Falling back to num_envs kwarg.")
                    env = gym.make(args.task, num_envs=args.robot_amount)
            except Exception as e:
                print(f"[WARN] Gym registry failed: {e}. Falling back to manual ctor.")

        if env is None:
            # Manual ctor for Go2 (or fallback when gym failed)
            from isaaclab.envs import ManagerBasedRLEnv
            if args.robot == "g1":
                try:
                    from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import G1RoughEnvCfg
                except Exception:
                    raise RuntimeError("G1RoughEnvCfg import failed; ensure isaaclab_tasks is installed.")
                cfg = G1RoughEnvCfg()
                exp_name_default = "g1_rough"
            elif args.robot == "go2":
                try:
                    from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import UnitreeGo2RoughEnvCfg
                except Exception:
                    raise RuntimeError("UnitreeGo2RoughEnvCfg import failed; ensure isaaclab_tasks is installed.")
                cfg = UnitreeGo2RoughEnvCfg()
                exp_name_default = "unitree_go2_rough"
            else:
                raise NotImplementedError(f"Unknown robot type: {args.robot}")

            cfg.scene.num_envs = args.robot_amount
            print(f"[INFO] Creating manual env (robot={args.robot}, num_envs={args.robot_amount})")
            env = ManagerBasedRLEnv(cfg)

        # --- Always wrap for RSL-RL ---
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        env = RslRlVecEnvWrapper(env)
        print("[INFO] Wrapped env with isaaclab_rl.rsl_rl.RslRlVecEnvWrapper.")

        # Agent cfg + checkpoint
        from agent_cfg import unitree_go2_agent_cfg, unitree_g1_agent_cfg
        agent_cfg = dict(unitree_g1_agent_cfg if args.robot == "g1" else unitree_go2_agent_cfg)
        exp_name = args.experiment_name or (exp_name_default if 'exp_name_default' in locals()
                                            else ("g1_rough" if args.robot == "g1" else "unitree_go2_rough"))
        ckpt_path = _find_checkpoint(exp_name, load_run=args.load_run, load_checkpoint=args.load_checkpoint)
        if not ckpt_path:
            raise RuntimeError("checkpoint_missing")

        # Sanity check: compare env dims vs checkpoint dims
        env_num_obs = getattr(env, "num_obs", None) or (env.observation_space.shape[0] if hasattr(env, "observation_space") else None)
        env_num_actions = getattr(env, "num_actions", None) or (env.action_space.shape[0] if hasattr(env, "action_space") else None)
        try:
            ckpt_blob = torch.load(ckpt_path, map_location="cpu")
            msd = ckpt_blob.get("model_state_dict", {})
            ckpt_num_actions = int(msd["std"].shape[0]) if "std" in msd else None
            ckpt_num_obs = int(msd["actor.0.weight"].shape[1]) if "actor.0.weight" in msd else None
            print(f"[INFO] Env dims: obs={env_num_obs}, act={env_num_actions} | Checkpoint dims: obs={ckpt_num_obs}, act={ckpt_num_actions}")
            if (ckpt_num_obs and env_num_obs and ckpt_num_obs != env_num_obs) or \
               (ckpt_num_actions and env_num_actions and ckpt_num_actions != env_num_actions):
                raise RuntimeError(
                    f"Checkpoint/env dimension mismatch. "
                    f"Env(obs={env_num_obs}, act={env_num_actions}) vs Ckpt(obs={ckpt_num_obs}, act={ckpt_num_actions}). "
                    f"Run with matching robot/task/config or pick the corresponding checkpoint."
                )
        except Exception as _e:
            # Non-fatal; OnPolicyRunner.load will still give a detailed error if incompatible
            print(f"[WARN] Pre-flight checkpoint shape check: {_e}")

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Loading checkpoint: {ckpt_path} (device={device})")
        from rsl_rl.runners import OnPolicyRunner  # type: ignore
        runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=device)
        runner.load(ckpt_path)
        policy = runner.get_inference_policy(device=getattr(env, "device", device))


        # Reset and play
        try:
            obs, _ = env.get_observations()
        except Exception:
            obs, _ = env.reset()

        print("[INFO] Starting inference loop...")
        while simulation_app.is_running():
            with torch.inference_mode():
                actions = policy(obs)
                step_out = env.step(actions)
                obs = step_out[0] if isinstance(step_out, tuple) and len(step_out) > 0 else step_out

        try:
            env.close()
        except Exception:
            pass
        print("[INFO] RL session ended.")
        return

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    except Exception as e:
        # Any failure: fall back to a minimal stage
        if str(e) != "checkpoint_missing":
            print(f"[WARN] RL path unavailable: {e}")
            traceback.print_exc()

    # Smoke fallback
    try:
        _create_minimal_stage()
        while simulation_app.is_running():
            simulation_app.update()
    finally:
        try:
            simulation_app.close()
        except Exception:
            pass
        print("[INFO] Isaac Sim closed.")


if __name__ == "__main__":
    run_sim()