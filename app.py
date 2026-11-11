"""Десктопное приложение для распознавания одного лица."""

import argparse
import logging
from threading import Thread
from time import monotonic
import tkinter as tk

import cv2
import numpy as np

from recognition import Camera, Enrollment, FRAME_TIMEOUT, MATCH_THRESHOLD, PROFILE
from recognition import load_profile, save_profile


TITLE = "Распознавание лица"
BG = "#0d141f"
PANEL = "#151f2d"
PREVIEW = "#090f19"
BORDER = "#28384b"
TEXT = "#f2f6fa"
MUTED = "#a7b8cb"
BLUE = "#90c8ff"
GREEN = "#61ecac"
AMBER = "#ffc98a"
DEFAULT_NOTE = "Профиль лица хранится на этом устройстве"


def label(parent, text, size=11, color=TEXT, background=PANEL, bold=False, **kwargs):
    return tk.Label(
        parent, text=text, fg=color, bg=background, bd=0,
        font=("Segoe UI", size, "bold" if bold else "normal"), **kwargs,
    )


class FaceApp:
    def __init__(self, root, camera_index):
        self.root = root
        self.camera_index = camera_index
        self.camera = None
        self.worker = None
        self.image = None
        self.enrollment = None
        self.matches = 0
        self.last_frame_at = None
        self.status = None
        self.ready = False
        self.camera_state = None
        note = DEFAULT_NOTE
        try:
            self.profile = load_profile(PROFILE)
        except (OSError, ValueError, EOFError):
            self.profile = None
            note = "Профиль не удалось прочитать. Запишите лицо заново."

        root.title(TITLE)
        root.geometry("1100x675")
        root.minsize(850, 560)
        root.configure(bg=BG)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<Escape>", self.cancel_enrollment)
        self.build_window(note)
        self.start_camera()
        self.poll_camera()

    def build_window(self, note):
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=30, pady=(22, 18))
        label(header, TITLE, size=25, background=BG, bold=True).pack(side="left")
        self.camera_label = label(header, "", color=MUTED, background=BG)
        self.camera_label.pack(side="right", padx=(10, 0))
        self.camera_light = tk.Canvas(header, width=10, height=10, bg=BG, highlightthickness=0)
        self.camera_light.pack(side="right")
        self.camera_dot = self.camera_light.create_oval(1, 1, 9, 9, fill=MUTED, outline="")

        footer = tk.Frame(self.root, bg=BG)
        footer.pack(side="bottom", fill="x", padx=30, pady=20)
        self.action = tk.Button(
            footer, command=self.on_action, font=("Segoe UI", 12, "bold"),
            bg=BLUE, fg=PREVIEW, activebackground="#bbddff", activeforeground=PREVIEW,
            disabledforeground="#7a8da4", highlightbackground=BG, highlightcolor=BLUE,
            highlightthickness=2, bd=0, padx=21, pady=8,
        )
        self.action.pack(side="left")
        self.action.bind("<Return>", lambda _event: self.action.invoke())
        self.note = label(footer, note, color=MUTED, background=BG, wraplength=430, justify="right")
        self.note.pack(side="right", padx=(16, 0))

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=30)
        body.columnconfigure(0, weight=6, uniform="panel")
        body.columnconfigure(1, weight=4, uniform="panel")
        body.rowconfigure(0, weight=1)
        camera_panel = tk.Frame(body, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        camera_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        result_panel = tk.Frame(body, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        result_panel.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        label(camera_panel, "Веб-камера", bold=True).pack(anchor="w", padx=18, pady=13)
        label(result_panel, "Результат", bold=True).pack(anchor="w", padx=18, pady=13)

        self.preview = tk.Canvas(camera_panel, bg=PREVIEW, highlightthickness=0)
        self.preview.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        self.preview_image = self.preview.create_image(0, 0)
        self.preview_message = self.preview.create_text(
            0, 0, fill=MUTED, justify="center", font=("Segoe UI", 12),
        )
        self.preview.bind("<Configure>", self.center_preview)

        result = tk.Frame(result_panel, bg=PANEL)
        result.pack(fill="both", expand=True, padx=18, pady=(0, 14))
        content = tk.Frame(result, bg=PANEL)
        content.place(relx=0.5, rely=0.5, relwidth=1, anchor="center")
        self.icon = tk.Canvas(content, bg=PANEL, width=192, height=192, highlightthickness=0)
        self.icon.pack()
        self.result_title = label(content, "", size=21, bold=True, wraplength=270)
        self.result_title.pack(fill="x", pady=(10, 6))
        self.result_detail = label(content, "", color=MUTED, wraplength=265, justify="center")
        self.result_detail.pack(fill="x")
        self.progress = tk.Canvas(content, bg=PANEL, width=142, height=7, highlightthickness=0)
        self.progress.pack(pady=(16, 0))

    def center_preview(self, event):
        center = (event.width // 2, event.height // 2)
        self.preview.coords(self.preview_image, *center)
        self.preview.coords(self.preview_message, *center)
        self.preview.itemconfigure(self.preview_message, width=max(1, event.width - 34))

    def set_camera_state(self, state):
        if self.camera_state == state:
            return
        self.camera_state = state
        text, color = {
            "connecting": ("Подключение камеры", MUTED),
            "live": ("Камера включена", GREEN),
            "stalled": ("Ожидание кадра", AMBER),
            "failed": ("Камера недоступна", AMBER),
        }[state]
        self.ready = state == "live"
        self.camera_label.configure(text=text, fg=color)
        self.camera_light.itemconfigure(self.camera_dot, fill=color)
        if not self.ready:
            self.matches = 0
            self.preview.itemconfigure(self.preview_image, image="")
            self.image = None
        self.preview.itemconfigure(self.preview_message, text="" if self.ready else text)
        self.update_action()

    def update_action(self):
        if self.camera_state == "failed":
            text, enabled = "Повторить подключение", True
        elif self.enrollment is not None:
            text, enabled = "Отменить запись", True
        else:
            text = "Записать лицо" if self.profile is None else "Записать заново"
            enabled = self.ready
        self.action.configure(
            text=text, state="normal" if enabled else "disabled",
            bg=BLUE if enabled else BORDER, cursor="hand2" if enabled else "",
        )

    def show_result(self, kind, title, detail, progress=None):
        status = (kind, title, detail, progress)
        if self.status == status:
            return
        self.status = status
        self.result_title.configure(text=title)
        self.result_detail.configure(text=detail)
        self.icon.delete("all")
        self.progress.delete("all")

        if kind == "match":
            self.icon.create_oval(6, 6, 186, 186, fill="#113a31", outline="#24654f", width=2)
            self.icon.create_line(
                40, 98, 77, 135, 151, 61,
                fill=GREEN, width=15, capstyle="round", joinstyle="round",
            )
        else:
            color = AMBER if kind == "alert" else BLUE
            self.icon.create_oval(6, 6, 186, 186, fill="#192a3c", outline=BORDER, width=2)
            for points in (
                (57, 79, 57, 57, 79, 57), (113, 57, 135, 57, 135, 79),
                (135, 113, 135, 135, 113, 135), (79, 135, 57, 135, 57, 113),
            ):
                self.icon.create_line(*points, fill=color, width=4, capstyle="round", joinstyle="round")
            self.icon.create_oval(82, 82, 110, 110, outline=color, width=3)
        if progress is not None:
            for i in range(Enrollment.SAMPLE_COUNT):
                self.progress.create_rectangle(
                    i * 29, 0, i * 29 + 26, 7, fill=BLUE if i < progress else BORDER, outline="",
                )

    def start_camera(self):
        self.set_camera_state("connecting")
        self.show_result("wait", "Подключаем камеру", "Подождите несколько секунд.")
        if self.worker is not None and self.worker.is_alive():
            self.root.after(50, self.start_camera)
            return
        self.camera = Camera(self.camera_index)
        self.last_frame_at = None
        self.worker = Thread(target=self.camera.run, daemon=True)
        self.worker.start()

    def poll_camera(self):
        now = monotonic()
        if self.enrollment is not None and now - self.enrollment.started_at >= Enrollment.DURATION:
            self.end_enrollment("Время записи истекло. Попробуйте ещё раз при хорошем освещении.")

        update = self.camera.updates.pop() if self.camera.updates else None
        if isinstance(update, str):
            self.end_enrollment()
            self.set_camera_state("failed")
            self.show_result("alert", "Камера недоступна", update)
        elif update is not None and now - update.captured_at < FRAME_TIMEOUT:
            if self.last_frame_at is None or update.captured_at - self.last_frame_at >= FRAME_TIMEOUT:
                self.matches = 0
            self.last_frame_at = update.captured_at
            self.set_camera_state("live")
            self.draw_frame(update.image)
            self.process_frame(update)
        elif self.ready and now - self.last_frame_at >= FRAME_TIMEOUT:
            self.set_camera_state("stalled")
            self.show_result("wait", "Ожидаем кадр", "Проверка продолжится, когда камера ответит.")
        self.root.after(30, self.poll_camera)

    def draw_frame(self, image):
        height, width = image.shape[:2]
        scale = min(self.preview.winfo_width() / width, self.preview.winfo_height() / height)
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        ppm = f"P6\n{size[0]} {size[1]}\n255\n".encode("ascii") + rgb.tobytes()
        self.image = tk.PhotoImage(data=ppm, format="PPM")
        self.preview.itemconfigure(self.preview_image, image=self.image)

    def on_action(self):
        if self.camera_state == "failed":
            self.start_camera()
        elif self.enrollment is not None:
            self.cancel_enrollment()
        elif self.ready:
            self.note.configure(text=DEFAULT_NOTE)
            self.matches = 0
            self.enrollment = Enrollment(monotonic())
            self.update_action()
            self.show_result("wait", "Записываем лицо", "Смотрите в камеру.", 0)

    def cancel_enrollment(self, _event=None):
        if self.enrollment is not None:
            self.end_enrollment("Запись отменена.")
            if self.ready:
                self.show_result("wait", "Смотрите в камеру", "Останьтесь в кадре.")

    def end_enrollment(self, note=None):
        self.enrollment = None
        self.update_action()
        if note is not None:
            self.note.configure(text=note)

    def process_frame(self, frame):
        if frame.faces > 1:
            hint = "Останьтесь в кадре одни."
        elif frame.faces == 0:
            hint = "Расположите лицо перед камерой."
        elif frame.feature is None:
            hint = "Подвиньтесь ближе и держите лицо целиком в кадре."
        else:
            hint = "Смотрите в камеру. Можно слегка повернуть голову."

        if self.enrollment is not None:
            feature = self.enrollment.add(frame)
            if feature is None:
                self.show_result("wait", "Записываем лицо", hint, len(self.enrollment.samples))
                return
            try:
                save_profile(PROFILE, feature)
            except OSError:
                logging.exception("Не удалось сохранить профиль")
                self.end_enrollment("Не удалось сохранить профиль. Попробуйте ещё раз.")
                return
            self.profile = feature
            self.end_enrollment("Профиль сохранён на этом устройстве.")

        if self.profile is None:
            self.show_result("wait", "Запишите своё лицо", "Нажмите «Записать лицо», когда будете в кадре.")
        elif frame.feature is None:
            self.matches = 0
            self.show_result("wait", "Смотрите в камеру", hint)
        elif np.dot(self.profile, frame.feature) < MATCH_THRESHOLD:
            self.matches = 0
            self.show_result("alert", "Лицо не совпало", "Посмотрите прямо в камеру при хорошем освещении.")
        else:
            self.matches += 1
            if self.matches >= 3:
                self.show_result("match", "Лицо распознано", "")
            else:
                self.show_result("wait", "Проверяем лицо", "Останьтесь в кадре.")

    def close(self):
        self.camera.stop.set()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=TITLE)
    parser.add_argument("--camera", type=int, default=0, help="Индекс камеры, начиная с 0")
    args = parser.parse_args()
    if args.camera < 0:
        parser.error("Индекс камеры должен быть неотрицательным")
    root = tk.Tk()
    app = FaceApp(root, args.camera)
    try:
        root.mainloop()
    finally:
        app.camera.stop.set()
        app.worker.join(timeout=1)


if __name__ == "__main__":
    main()
