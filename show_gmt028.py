"""GMT028-09 (ST7789, 4-wire SPI) single-screen wiring test.

Run --preview first on any computer. Hardware mode runs on a Raspberry Pi and
keeps the backlight enabled until Ctrl-C.
"""

from __future__ import annotations

import argparse
import time

WIDTH = 240
HEIGHT = 320


def make_frame(text: str):
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default()
    bounds = font.getbbox(text)
    text_width = bounds[2] - bounds[0]
    text_height = bounds[3] - bounds[1]
    if not text_width or not text_height or text_width > WIDTH - 16:
        raise ValueError("Text is empty or too long for this display")
    scale = min(3, (WIDTH - 16) // text_width)
    label = Image.new("RGB", (text_width, text_height), "white")
    ImageDraw.Draw(label).text((-bounds[0], -bounds[1]), text, font=font, fill="black")
    label = label.resize((text_width * scale, text_height * scale), Image.Resampling.NEAREST)
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    image.paste(label, ((WIDTH - label.width) // 2, (HEIGHT - label.height) // 2))
    return image


def rgb565_bytes(image) -> bytes:
    result = bytearray(WIDTH * HEIGHT * 2)
    for index, (red, green, blue) in enumerate(image.getdata()):
        pixel = ((red >> 3) << 11) | ((green >> 2) << 5) | (blue >> 3)
        result[2 * index] = pixel >> 8
        result[2 * index + 1] = pixel & 0xFF
    return bytes(result)


def show_on_screen(image, dc_pin: int, reset_pin: int, backlight_pin: int) -> None:
    import spidev
    from gpiozero import OutputDevice

    dc = OutputDevice(dc_pin, initial_value=False)
    reset = OutputDevice(reset_pin, initial_value=True)
    backlight = OutputDevice(backlight_pin, initial_value=False)
    spi = spidev.SpiDev()
    try:
        spi.open(0, 0)  # SPI0 CE0, mode 0, 8-bit words.
        spi.mode = 0
        spi.max_speed_hz = 8_000_000

        def send(command: int, data: bytes = b"") -> None:
            dc.off()
            spi.writebytes2([command])
            if data:
                dc.on()
                spi.writebytes2(data)

        reset.off()
        time.sleep(0.02)
        reset.on()
        time.sleep(0.15)
        send(0x01)  # SWRESET
        time.sleep(0.15)
        send(0x11)  # SLPOUT
        time.sleep(0.15)
        send(0x3A, b"\x55")  # COLMOD: RGB565
        send(0x36, b"\x00")  # MADCTL: portrait
        send(0x2A, (0).to_bytes(2, "big") + (WIDTH - 1).to_bytes(2, "big"))  # CASET
        send(0x2B, (0).to_bytes(2, "big") + (HEIGHT - 1).to_bytes(2, "big"))  # RASET
        send(0x29)  # DISPON
        time.sleep(0.02)
        send(0x2C)  # RAMWR
        dc.on()
        pixels = rgb565_bytes(image)
        for offset in range(0, len(pixels), 4096):
            spi.writebytes2(pixels[offset : offset + 4096])
        backlight.on()
        print("Screen updated. Press Ctrl-C to turn off the backlight.")
        while True:
            time.sleep(1)
    finally:
        backlight.off()
        spi.close()
        dc.close()
        reset.close()
        backlight.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="123+123+123")
    parser.add_argument("--preview", metavar="PNG", help="save a preview without opening SPI/GPIO")
    parser.add_argument("--dc", type=int, default=25, help="BCM GPIO number, default 25")
    parser.add_argument("--reset", type=int, default=24, help="BCM GPIO number, default 24")
    parser.add_argument("--backlight", type=int, default=23, help="BCM GPIO number, default 23")
    args = parser.parse_args()
    image = make_frame(args.text)
    if args.preview:
        image.save(args.preview)
        print(f"Preview saved: {args.preview}")
        return
    try:
        show_on_screen(image, args.dc, args.reset, args.backlight)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
