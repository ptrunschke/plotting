from __future__ import annotations
from pathlib import Path
from PIL import Image
import re


def glob_to_regex(glob: str) -> str:
    def regex_escape(character: str) -> str:
        return rf"\{character}" if character in "$^+.()|" else character

    pattern = ""
    inGroup = False
    starDetected = 0
    for token in glob:
        if token == "*":
            starDetected += 1
            continue

        if starDetected == 1:
            pattern += "([^/]*)"
            starDetected = 0
        elif starDetected == 2:
            pattern += ".*?"
            starDetected = 0
        if starDetected > 2:
            raise ValueError("more than two consecutive stars detected")

        if token == "?":
            pattern += "[^/]"
        elif token == "{":
            assert not inGroup
            pattern += "("
            inGroup = True
        elif token == "}":
            assert inGroup
            pattern += ")"
            inGroup = False
        elif token == "," and inGroup:
            pattern += "|"
        else:
            pattern += regex_escape(token)

    return pattern


def create_gif(
    pattern: str, gifPath: str = "animation.gif", duration: float = 100.0, loop: int = 0
) -> None:
    """Create a GIF from the images matching the prescibed pattern.

    Parameters
    ----------
    pattern: str
        Pattern to match.
        It is assumed that the final star matches a numeric value.
    gifName: str
        Path of the output file.
    duration: float, default=100
        Duration for each frame in ms.
    loop: int, default=0
        Number of times gif is looped. Zero means loop forever.
    """
    rx = re.compile(glob_to_regex(pattern))

    def sort_key(path: Path) -> float:
        matches = rx.findall(str(path))
        assert len(matches) == 1
        matches = matches[0]
        return float(matches if isinstance(matches, str) else matches[-1])

    images = sorted(Path(".").glob(pattern), key=sort_key)
    frames = [Image.open(str(image)) for image in images]
    frames.pop(0).save(
        gifPath,
        format="GIF",
        append_images=frames,
        save_all=True,
        duration=duration,
        loop=loop,
        transparency=0,
        disposal=2,
    )


if __name__ == "__main__":
    import argparse

    descr = """Turn a collection of images into a GIF."""
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument("PATTERN", type=str, help="glob pattern for the image files")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="animation.gif",
        help="path of the output file",
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=float,
        default=100,
        help="duration for each frame in ms (default 100)",
    )
    parser.add_argument(
        "-l",
        "--loop",
        type=int,
        default=0,
        help="number of times the GIF is looped (default 0, loop forever)",
    )
    args = parser.parse_args()
    create_gif(
        pattern=args.PATTERN,
        gifPath=args.output,
        duration=args.duration,
        loop=args.loop,
    )
