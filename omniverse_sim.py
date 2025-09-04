"""Play a trained checkpoint (RSL-RL) if available. No ROS2. Safe fallbacks."""
from __future__ import annotations

import argparse
import os
import sys
import time
import glob
import re
import traceback

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

def _resolve_experience_path(apps_dir: str | None, app_name: str, headless: bool) -> str | None:
    """Map --app to the right .kit in apps_dir."""
    name_map = {
        "python": "isaaclab.python.kit",
        "python.rendering": "isaaclab.python.rendering.kit",
        "python.headless": "isaaclab.python.headless.kit",
        "python.headless.rendering": "isaaclab.python.headless.rendering.kit",
        "python.xr.openxr": "isaaclab.python.xr.openxr.kit",
        "python.xr.openxr.headless": "isaaclab.python.xr.openxr.headless.kit",
    }
    # default if user didn’t pass --app
    if app_name is None:
        app_name = "python.headless" if headless else "python"
    kit_file = name_map.get(app_name)
    if not kit_file or not apps_dir:
        return None
    kit_path = os.path.join(apps_dir, kit_file)
    return kit_path if os.path.isfile(kit_path) else None

def _get_sim_app(headless: bool, app_name: str | None = None):
    """Return SimulationApp for Isaac Sim 5.x or 2023.x (fallback) with Isaac Lab app experience on Windows."""
    # Try Isaac Sim 5.x
    try:
        from isaacsim import SimulationApp  # type: ignore
        apps_dir = _locate_apps_dir()
        exp = _resolve_experience_path(apps_dir, app_name, headless)
        if exp:
            print(f"[INFO] Using Isaac Lab experience: {exp}")
            return SimulationApp({"headless": headless, "experience": exp})
        # Fallback to default experience if not found
        print("[WARN] Isaac Lab apps dir or .kit not found; using default experience.")
        return SimulationApp({"headless": headless})
    except Exception:
        # Legacy Isaac Sim 2023.x
        from omni.isaac.kit import SimulationApp  # type: ignore
        return SimulationApp({"headless": headless})

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
    # Isaac Lab AppLauncher로 SimulationApp 시작 (경험 파일/렌더링 모드 자동 선택)
    print("[INFO] Launching Isaac Sim via Isaac Lab AppLauncher ...")
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Try full RL playback path (requires your env + rsl_rl). On failure, fallback to smoke.
    try:
        _ensure_lab_registry()
        import torch  # type: ignore
        from rsl_rl.runners import OnPolicyRunner  # type: ignore

        env = None
        # create env via Isaac Lab 2.2 registry
        if args.task.startswith("Isaac-"):
            import gymnasium as gym
            print(f"[INFO] Creating gym env: {args.task} (num_envs={args.robot_amount})")
            try:
                env = gym.make(args.task, num_envs=args.robot_amount)
            except Exception as e:
                print(f"[WARN] Gym registry failed: {e}. Falling back to manual ctor.")

        if env is None:
            # Manual ctor with Isaac Lab cfg
            from isaaclab.envs import ManagerBasedRLEnv
            if args.robot == "g1":
                try:
                    from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import G1RoughEnvCfg
                except Exception:
                    from isaaclab_tasks.manager_based.locomotion.velocity.config.unitree.g1.rough_env_cfg import G1RoughEnvCfg  # type: ignore
                cfg = G1RoughEnvCfg()
                exp_name_default = "g1_rough"
            else:
                try:
                    from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import UnitreeGo2RoughEnvCfg
                except Exception:
                    from isaaclab_tasks.manager_based.locomotion.velocity.config.unitree.go2.rough_env_cfg import UnitreeGo2RoughEnvCfg  # type: ignore
                cfg = UnitreeGo2RoughEnvCfg()
                exp_name_default = "unitree_go2_rough"

            cfg.scene.num_envs = args.robot_amount
            print(f"[INFO] Creating manual env (robot={args.robot}, num_envs={args.robot_amount})")
            base_env = ManagerBasedRLEnv(cfg)

            # Wrap with RSL-RL vectorized wrapper to provide get_observations API
            try:
                from isaaclab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper
                env = RslRlVecEnvWrapper(base_env)
                print("[INFO] Wrapped env with RslRlVecEnvWrapper.")
            except Exception as e:
                print(f"[WARN] Failed to wrap env for RSL-RL ({e}). Using base env; some runners may not work.")
                env = base_env

        # # Build env config
        # env_cfg = EnvCfg()
        # # Best-effort: set number of envs if attribute path exists
        # try:
        #     env_cfg.scene.num_envs = args.robot_amount  # type: ignore[attr-defined]
        # except Exception:
        #     pass

        # # Create env via gym registry (must be registered elsewhere)
        # print(f"[INFO] Creating gym env: {args.task}")
        # try:
        #     env = gym.make(args.task, cfg=env_cfg)
        # except Exception as e:
        #     print(f"[WARN] gym.make failed for '{args.task}': {e}")
        #     raise

        # If you had a VecEnv wrapper in Orbit, skip unless available in your setup.
        # from omni.isaac.orbit_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper
        # env = RslRlVecEnvWrapper(env)

        # Agent cfg + checkpoint
        from agent_cfg import unitree_go2_agent_cfg, unitree_g1_agent_cfg
        agent_cfg = dict(unitree_g1_agent_cfg if args.robot == "g1" else unitree_go2_agent_cfg)
        exp_name = args.experiment_name or (exp_name_default if 'exp_name_default' in locals() else
                                            ("g1_rough" if args.robot == "g1" else "unitree_go2_rough"))
        ckpt_path = _find_checkpoint(exp_name, load_run=args.load_run, load_checkpoint=args.load_checkpoint)
        if not ckpt_path:
            raise RuntimeError("checkpoint_missing")

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Loading checkpoint: {ckpt_path} (device={device})")
        runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=device)
        runner.load(ckpt_path)
        policy = runner.get_inference_policy(device=getattr(env, "device", device))


        # Reset and play
        try:
            obs, _ = env.get_observations()
        except Exception:
            # Generic reset fallback
            obs, _ = env.reset()

        print("[INFO] Starting inference loop...")
        while simulation_app.is_running():
            with torch.inference_mode():
                actions = policy(obs)
                step_out = env.step(actions)
                # Support both (obs, ...) or plain obs
                if isinstance(step_out, tuple) and len(step_out) > 0:
                    obs = step_out[0]
                else:
                    obs = step_out

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