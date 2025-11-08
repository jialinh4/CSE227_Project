import typer
from typing import Optional
import re, yaml
from .pipeline import EditConfig, process_video
from .segmentation import segment_video as run_segmentation

app = typer.Typer(help="vaderx video tools", add_completion=False)

def parse_size(s: str):
    import re
    m = re.match(r"^(\d+)x(\d+)$", s.strip())
    if not m:
        raise typer.BadParameter("Use WxH, e.g., 1280x720")
    return (int(m.group(1)), int(m.group(2)))

@app.command()
def edit(
    in_path: str = typer.Option(..., "--in_path", help="Input video path"),
    out_path: str = typer.Option(..., "--out_path", help="Output video path"),
    start: float = typer.Option(0.0, help="Start time (sec)"),
    end: float = typer.Option(-1.0, help="End time (sec), -1 till end"),
    resize: Optional[str] = typer.Option(None, help="WxH, e.g., 1280x720"),
    gray: bool = typer.Option(False, "--gray/--no-gray", help="Convert to grayscale"),
    config: Optional[str] = typer.Option(None, "--config", help="YAML config (overrides CLI)"),
):
    """Edit a single video with basic options."""
    if config:
        with open(config, "r") as f:
            cfg_yaml = yaml.safe_load(f)
        in_path = cfg_yaml.get("in_path", in_path)
        out_path = cfg_yaml.get("out_path", out_path)
        start = float(cfg_yaml.get("start", start))
        end   = float(cfg_yaml.get("end", end))
        resize = cfg_yaml.get("resize", resize)
        gray   = bool(cfg_yaml.get("gray", gray))

    size = parse_size(resize) if resize else None
    cfg = EditConfig(start_sec=start, end_sec=end, resize_to=size, gray=gray)
    process_video(in_path, out_path, cfg)
    typer.echo(f"✔ Wrote {out_path}")


@app.command()
def segment(
    in_path: str = typer.Option(..., "--in_path", help="Input video path"),
    mask_root: str = typer.Option("masks", "--mask_root", help="Destination root for masks"),
    video_id: Optional[str] = typer.Option(None, "--video_id", help="Optional override for mask folder name"),
    backend: str = typer.Option("auto", "--backend", help="Segmentation backend: auto|mediapipe|deeplabv3"),
    stride: int = typer.Option(1, "--stride", min=1, help="Process every Nth frame"),
    smooth_kernel: int = typer.Option(
        0, "--smooth_kernel", help="Optional Gaussian blur kernel (odd, >=3) for alpha smoothing"
    ),
):
    """Generate per-frame alpha masks for a video."""
    result = run_segmentation(
        in_path=in_path,
        mask_root=mask_root,
        backend=backend,
        video_id=video_id,
        stride=stride,
        smooth_kernel=smooth_kernel,
        progress=True,
    )
    typer.echo(
        f"✔ Wrote {result.masks_written} masks to {result.mask_dir} "
        f"(backend={result.backend})"
    )

if __name__ == "__main__":
    app()
