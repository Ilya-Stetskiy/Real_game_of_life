import math
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageTk


DEFAULT_IMAGE_PATH = (
    Path(__file__).resolve().parent
    / "H2BmCherry_timelapse_60h"
    / "H2BmCherry_timelapse_60h.tif"
)
DEFAULT_MASK_PATH = (
    Path(__file__).resolve().parent
    / "H2BmCherry_timelapse_60h"
    / "H2BmCherry_timelapse_60h_mask.tif"
)
PROCESSED_ARRAYS_DIR = Path(__file__).resolve().parent / "processed_arrays"


def get_divisor_presets(width: int, height: int) -> tuple[str, ...]:
    common_divisors = []
    limit = min(width, height)
    for value in range(1, limit + 1):
        if width % value == 0 and height % value == 0:
            common_divisors.append(value)

    preferred = [value for value in common_divisors if 8 <= value <= 256]
    if preferred:
        return tuple(str(value) for value in preferred)
    return tuple(str(value) for value in common_divisors)


def normalize_to_uint8(frame: np.ndarray) -> Image.Image:
    array = np.asarray(frame)
    if array.ndim > 2:
        array = np.squeeze(array)

    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    min_value = float(np.min(array))
    max_value = float(np.max(array))

    if math.isclose(max_value, min_value):
        normalized = np.zeros(array.shape, dtype=np.uint8)
    else:
        scaled = (array - min_value) / (max_value - min_value)
        normalized = np.clip(scaled * 255.0, 0, 255).astype(np.uint8)

    return Image.fromarray(normalized, mode="L")


def compute_grid_fill_ratios(mask_frame: np.ndarray, grid_size: int) -> np.ndarray:
    if grid_size <= 0:
        raise ValueError("grid_size must be positive.")

    binary = np.not_equal(np.asarray(mask_frame).squeeze(), 0)
    if binary.ndim != 2:
        raise ValueError(f"Expected a 2D mask frame, got shape {binary.shape}.")

    height, width = binary.shape
    row_starts = np.arange(0, height, grid_size)
    col_starts = np.arange(0, width, grid_size)
    row_sizes = np.diff(np.r_[row_starts, height]).astype(np.float32)
    col_sizes = np.diff(np.r_[col_starts, width]).astype(np.float32)

    counts = np.add.reduceat(binary.astype(np.float32), row_starts, axis=0)
    counts = np.add.reduceat(counts, col_starts, axis=1)
    return counts / (row_sizes[:, None] * col_sizes[None, :])


def compute_cell_array(mask_stack: "TiffStack", grid_size: int, zero_count: int) -> np.ndarray:
    frame_count = mask_stack.frame_count
    rows = math.ceil(mask_stack.height / grid_size)
    cols = math.ceil(mask_stack.width / grid_size)
    vector_size = zero_count + 1
    result = np.zeros((frame_count, rows, cols, vector_size), dtype=np.float32)

    for frame_index in range(frame_count):
        frame = np.asarray(mask_stack.get_frame(frame_index))
        result[frame_index, :, :, 0] = compute_grid_fill_ratios(frame, grid_size)

    return result


def create_value_overlay(
    data_2d: np.ndarray,
    image_width: int,
    image_height: int,
    grid_size: int,
) -> Image.Image:
    rows, cols = data_2d.shape
    image = Image.new("RGB", (image_width, image_height), "#101010")
    draw = ImageDraw.Draw(image)

    for row in range(rows):
        y0 = row * grid_size
        y1 = min(y0 + grid_size, image_height)
        for col in range(cols):
            x0 = col * grid_size
            x1 = min(x0 + grid_size, image_width)
            value = float(data_2d[row, col])
            intensity = int(round(max(0.0, min(1.0, value)) * 255))
            color = (intensity, 64, 255 - intensity)
            draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=color, outline="#1f1f1f")

            cell_w = x1 - x0
            cell_h = y1 - y0
    return image


def find_tiff_mask_pairs(root_dir: Path) -> list[tuple[Path, Path]]:
    image_candidates: dict[str, list[Path]] = {}
    mask_candidates: dict[str, list[Path]] = {}

    for path in root_dir.rglob("*"):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        if suffix not in {".tif", ".tiff"}:
            continue

        stem_lower = path.stem.lower()
        if stem_lower.endswith("_mask"):
            base_stem = path.stem[:-5]
            key = base_stem.lower()
            mask_candidates.setdefault(key, []).append(path)
        elif stem_lower.startswith("lblimg_"):
            base_stem = path.stem[7:]
            key = base_stem.lower()
            mask_candidates.setdefault(key, []).append(path)
        else:
            key = path.stem.lower()
            image_candidates.setdefault(key, []).append(path)

    pairs = []
    for key, image_paths in image_candidates.items():
        mask_paths = mask_candidates.get(key)
        if not mask_paths:
            continue

        if len(image_paths) == 1 and len(mask_paths) == 1:
            pairs.append((image_paths[0], mask_paths[0]))

    pairs.sort(key=lambda pair: (pair[0].stem.lower(), str(pair[0]).lower(), str(pair[1]).lower()))
    return pairs


class TiffStack:
    def __init__(self, path: str):
        self.path = path
        self._tiff = tifffile.TiffFile(path)
        self._series = self._tiff.series[0]
        self.shape = self._series.shape
        self.dtype = self._series.dtype

        if len(self.shape) < 2:
            raise ValueError(f"Unsupported TIFF shape: {self.shape}")

        if len(self.shape) == 2:
            self.frame_count = 1
            self.height, self.width = self.shape
        else:
            self.frame_count = int(self.shape[0])
            self.height = int(self.shape[-2])
            self.width = int(self.shape[-1])

    def get_frame(self, index: int) -> np.ndarray:
        if self.frame_count == 1:
            return self._series.asarray()
        return self._series.asarray(key=index)

    def close(self) -> None:
        self._tiff.close()


class DisplayPanel(ttk.Frame):
    def __init__(self, parent, title: str):
        super().__init__(parent, padding=6)
        self._photo = None
        self._last_image = None
        self._last_grid = None

        ttk.Label(self, text=title, font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.canvas = tk.Canvas(self, width=420, height=420, background="#111111", highlightthickness=1)
        self.canvas.configure(highlightbackground="#888888")
        self.canvas.pack(fill="both", expand=True, pady=(6, 0))
        self.canvas.bind("<Configure>", self._handle_resize)

    def show_array_frame(self, frame: np.ndarray, grid_size: int | None) -> None:
        self.show_pil_image(normalize_to_uint8(frame), grid_size, frame.shape[-1], frame.shape[-2])

    def show_pil_image(
        self,
        image: Image.Image,
        grid_size: int | None = None,
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> None:
        source = image.convert("RGB")
        source_width = image_width or source.width
        source_height = image_height or source.height
        self._last_image = source
        self._last_grid = (grid_size, source_width, source_height)

        canvas_width = max(self.canvas.winfo_width(), 1)
        canvas_height = max(self.canvas.winfo_height(), 1)
        scale = min(canvas_width / source_width, canvas_height / source_height)
        scale = max(scale, 0.05)

        display_width = max(1, int(source_width * scale))
        display_height = max(1, int(source_height * scale))
        resized = source.resize((display_width, display_height), Image.Resampling.NEAREST)

        self._photo = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")

        x0 = (canvas_width - display_width) // 2
        y0 = (canvas_height - display_height) // 2
        self.canvas.create_image(x0, y0, anchor="nw", image=self._photo)

        if grid_size and grid_size > 0:
            self._draw_grid(x0, y0, scale, grid_size, source_width, source_height)

    def _handle_resize(self, _event) -> None:
        if self._last_image is not None and self._last_grid is not None:
            grid_size, image_width, image_height = self._last_grid
            self.show_pil_image(self._last_image, grid_size, image_width, image_height)

    def _draw_grid(
        self,
        x0: int,
        y0: int,
        scale: float,
        grid_size: int,
        image_width: int,
        image_height: int,
    ) -> None:
        line_color = "#00e5ff"
        scaled_width = int(round(image_width * scale))
        scaled_height = int(round(image_height * scale))

        for x in range(grid_size, image_width, grid_size):
            xpos = x0 + int(round(x * scale))
            self.canvas.create_line(xpos, y0, xpos, y0 + scaled_height, fill=line_color, width=1)

        for y in range(grid_size, image_height, grid_size):
            ypos = y0 + int(round(y * scale))
            self.canvas.create_line(x0, ypos, x0 + scaled_width, ypos, fill=line_color, width=1)

        self.canvas.create_rectangle(
            x0,
            y0,
            x0 + scaled_width,
            y0 + scaled_height,
            outline=line_color,
            width=1,
        )


class TiffGridViewerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("TIFF Grid Viewer")
        self.root.geometry("1560x860")
        self.root.minsize(1200, 720)

        self.image_stack: TiffStack | None = None
        self.mask_stack: TiffStack | None = None
        self.generated_array: np.ndarray | None = None
        self.generated_array_path: Path | None = None

        self.current_frame = 0
        self.grid_size: int | None = None
        self.grid_presets = ("1",)

        self.image_path_var = tk.StringVar(value=str(DEFAULT_IMAGE_PATH if DEFAULT_IMAGE_PATH.exists() else ""))
        self.mask_path_var = tk.StringVar(value=str(DEFAULT_MASK_PATH if DEFAULT_MASK_PATH.exists() else ""))
        self.grid_preset_var = tk.StringVar(value="")
        self.grid_entry_var = tk.StringVar(value="")
        self.zero_count_var = tk.StringVar(value="0")
        self.save_batch_here_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Загрузите TIFF и mask TIFF.")
        self.frame_label_var = tk.StringVar(value="Кадр: - / -")
        self.array_info_var = tk.StringVar(value="Массив ещё не сгенерирован.")

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<Left>", lambda _event: self.step_frame(-1))
        self.root.bind("<Right>", lambda _event: self.step_frame(1))

        if self.image_path_var.get() and self.mask_path_var.get():
            self.load_stacks()

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)

        controls = ttk.Frame(main)
        controls.pack(fill="x")

        self._build_path_row(
            controls,
            row=0,
            label="TIFF",
            variable=self.image_path_var,
            browse_command=lambda: self.pick_file(self.image_path_var, "Выберите TIFF"),
        )
        self._build_path_row(
            controls,
            row=1,
            label="Mask TIFF",
            variable=self.mask_path_var,
            browse_command=lambda: self.pick_file(self.mask_path_var, "Выберите mask TIFF"),
        )

        actions = ttk.Frame(controls)
        actions.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        actions.columnconfigure(12, weight=1)

        ttk.Button(actions, text="Загрузить", command=self.load_stacks).grid(row=0, column=0, padx=(0, 8))
        ttk.Label(actions, text="Размер ячейки (px):").grid(row=0, column=1, padx=(0, 6))

        self.preset_box = ttk.Combobox(
            actions,
            textvariable=self.grid_preset_var,
            values=self.grid_presets,
            width=8,
            state="readonly",
        )
        self.preset_box.grid(row=0, column=2, padx=(0, 8))
        self.preset_box.bind("<<ComboboxSelected>>", self._copy_preset_to_entry)

        ttk.Entry(actions, textvariable=self.grid_entry_var, width=8).grid(row=0, column=3, padx=(0, 8))
        ttk.Button(actions, text="Построить сетку", command=self.apply_grid).grid(row=0, column=4, padx=(0, 10))

        ttk.Label(actions, text="Число нулей N:").grid(row=0, column=5, padx=(0, 6))
        ttk.Entry(actions, textvariable=self.zero_count_var, width=8).grid(row=0, column=6, padx=(0, 8))
        ttk.Button(actions, text="Сгенерировать массив", command=self.generate_array).grid(
            row=0, column=7, padx=(0, 10)
        )
        ttk.Button(actions, text="Обработать группу", command=self.open_batch_processing_dialog).grid(
            row=0, column=8, padx=(0, 10)
        )
        ttk.Button(actions, text="Скрыть сетку", command=self.clear_grid).grid(row=0, column=9, padx=(0, 10))
        ttk.Button(actions, text="<<", command=lambda: self.step_frame(-1)).grid(row=0, column=10, padx=(0, 4))
        ttk.Button(actions, text=">>", command=lambda: self.step_frame(1)).grid(row=0, column=11, padx=(0, 8))
        ttk.Label(actions, textvariable=self.frame_label_var).grid(row=0, column=12, sticky="e")
        ttk.Label(actions, textvariable=self.array_info_var).grid(row=0, column=13, sticky="e")

        self.frame_slider = ttk.Scale(main, from_=0, to=0, orient="horizontal", command=self.on_slider_change)
        self.frame_slider.pack(fill="x", pady=(10, 10))

        panels = ttk.Frame(main)
        panels.pack(fill="both", expand=True)
        for column in range(3):
            panels.columnconfigure(column, weight=1)
        panels.rowconfigure(0, weight=1)

        self.image_panel = DisplayPanel(panels, "Исходный TIFF")
        self.image_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self.mask_panel = DisplayPanel(panels, "Mask TIFF")
        self.mask_panel.grid(row=0, column=1, sticky="nsew", padx=5)
        self.values_panel = DisplayPanel(panels, "Заполненность ячеек")
        self.values_panel.grid(row=0, column=2, sticky="nsew", padx=(5, 0))

        status = ttk.Label(main, textvariable=self.status_var, anchor="w")
        status.pack(fill="x", pady=(8, 0))

    @staticmethod
    def _build_path_row(parent, row: int, label: str, variable: tk.StringVar, browse_command) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=8, pady=4)
        ttk.Button(parent, text="Обзор...", command=browse_command).grid(row=row, column=2, pady=4)
        parent.columnconfigure(1, weight=1)

    def pick_file(self, variable: tk.StringVar, title: str) -> None:
        selected = filedialog.askopenfilename(
            title=title,
            filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")],
        )
        if selected:
            variable.set(selected)

    def load_stacks(self) -> None:
        image_path = self.image_path_var.get().strip()
        mask_path = self.mask_path_var.get().strip()

        if not image_path or not mask_path:
            messagebox.showerror("Не хватает файлов", "Укажите путь и к TIFF, и к mask TIFF.")
            return

        try:
            new_image_stack = TiffStack(image_path)
            new_mask_stack = TiffStack(mask_path)
        except Exception as exc:
            messagebox.showerror("Ошибка загрузки", str(exc))
            return

        try:
            self._validate_stacks(new_image_stack, new_mask_stack)
        except Exception as exc:
            new_image_stack.close()
            new_mask_stack.close()
            messagebox.showerror("Несовпадение данных", str(exc))
            return

        self._close_stacks()
        self.image_stack = new_image_stack
        self.mask_stack = new_mask_stack
        self.generated_array = None
        self.generated_array_path = None
        self.array_info_var.set("Массив ещё не сгенерирован.")
        self._update_grid_presets()
        self.current_frame = 0
        self.frame_slider.configure(to=self.image_stack.frame_count - 1)
        self.frame_slider.set(0)
        self.status_var.set(
            f"Загружено: {Path(image_path).name} и {Path(mask_path).name}. "
            f"Кадров: {self.image_stack.frame_count}, размер: {self.image_stack.width}x{self.image_stack.height}."
        )
        self.apply_grid(silent=True)
        self.render_current_frame()

    @staticmethod
    def _validate_stacks(image_stack: TiffStack, mask_stack: TiffStack) -> None:
        if image_stack.frame_count != mask_stack.frame_count:
            raise ValueError("Количество кадров у TIFF и mask TIFF отличается.")
        if (image_stack.width, image_stack.height) != (mask_stack.width, mask_stack.height):
            raise ValueError("Размер кадров у TIFF и mask TIFF отличается.")

    def _copy_preset_to_entry(self, _event=None) -> None:
        self.grid_entry_var.set(self.grid_preset_var.get())
        self.apply_grid(silent=True)

    def _update_grid_presets(self) -> None:
        if self.image_stack is None:
            self.grid_presets = ("1",)
        else:
            self.grid_presets = get_divisor_presets(self.image_stack.width, self.image_stack.height)

        self.preset_box.configure(values=self.grid_presets)
        preferred_default = next((value for value in self.grid_presets if int(value) >= 8), self.grid_presets[0])
        self.grid_preset_var.set(preferred_default)
        self.grid_entry_var.set(preferred_default)

    def apply_grid(self, silent: bool = False) -> None:
        value = self.grid_entry_var.get().strip()
        if not value:
            self.grid_size = None
            if not silent:
                self.status_var.set("Сетка скрыта: размер ячейки не задан.")
            self.render_current_frame()
            return

        try:
            grid_size = int(value)
        except ValueError:
            if not silent:
                messagebox.showerror("Некорректный размер", "Размер ячейки должен быть целым числом пикселей.")
            return

        if grid_size <= 0:
            if not silent:
                messagebox.showerror("Некорректный размер", "Размер ячейки должен быть больше нуля.")
            return

        self.grid_size = grid_size
        if not silent:
            if self.image_stack is not None:
                cols = math.ceil(self.image_stack.width / grid_size)
                rows = math.ceil(self.image_stack.height / grid_size)
                rem_x = self.image_stack.width % grid_size
                rem_y = self.image_stack.height % grid_size
                self.status_var.set(
                    "Сетка включена. "
                    f"Размер ячейки: {grid_size} px. Ячеек: {cols} x {rows}. "
                    f"Остаток по краям: {rem_x} px x {rem_y} px."
                )
            else:
                self.status_var.set(f"Сетка включена. Размер ячейки: {grid_size} px.")
        self.render_current_frame()

    def clear_grid(self) -> None:
        self.grid_size = None
        self.status_var.set("Сетка скрыта.")
        self.render_current_frame()

    def generate_array(self) -> None:
        if self.mask_stack is None or self.image_stack is None:
            messagebox.showerror("Нет данных", "Сначала загрузите TIFF и mask TIFF.")
            return

        if self.grid_size is None:
            messagebox.showerror("Нет сетки", "Сначала задайте размер ячейки и постройте сетку.")
            return

        try:
            zero_count = int(self.zero_count_var.get().strip())
        except ValueError:
            messagebox.showerror("Некорректный размер", "Число нулей N должно быть целым числом.")
            return

        if zero_count < 0:
            messagebox.showerror("Некорректный размер", "Число нулей N должно быть не меньше 0.")
            return

        self.status_var.set("Идёт генерация массива. Это может занять некоторое время.")
        self.root.update_idletasks()

        try:
            generated = compute_cell_array(self.mask_stack, self.grid_size, zero_count)
        except Exception as exc:
            messagebox.showerror("Ошибка генерации", str(exc))
            return

        self.generated_array = generated
        output_path = self._suggest_output_path()
        np.save(output_path, self.generated_array)
        self.generated_array_path = output_path

        frames, rows, cols, depth = self.generated_array.shape
        self.array_info_var.set(f"Массив: {frames} x {rows} x {cols} x {depth}")
        self.status_var.set(f"Массив сгенерирован и сохранён: {output_path.name}")
        self.render_current_frame()

    def _suggest_output_path(
        self,
        mask_path: str | Path | None = None,
        save_to_processed_dir: bool = False,
        batch_root_dir: Path | None = None,
    ) -> Path:
        if mask_path is None and self.mask_stack is None:
            return Path(__file__).resolve().parent / "generated_cell_array.npy"

        base = Path(mask_path) if mask_path is not None else Path(self.mask_stack.path)
        stem = base.stem
        zero_count = self.zero_count_var.get().strip()
        file_name = f"{stem}_grid_{self.grid_size}_zeros_{zero_count}.npy"
        if save_to_processed_dir:
            target_dir = PROCESSED_ARRAYS_DIR
            if batch_root_dir is not None:
                relative_dir = base.parent.relative_to(batch_root_dir)
                target_dir = target_dir / relative_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            return target_dir / file_name
        return base.with_name(file_name)

    def _validate_zero_count(self) -> int | None:
        try:
            zero_count = int(self.zero_count_var.get().strip())
        except ValueError:
            messagebox.showerror("Некорректный размер", "Число нулей N должно быть целым числом.")
            return None

        if zero_count < 0:
            messagebox.showerror("Некорректный размер", "Число нулей N должно быть не меньше 0.")
            return None

        return zero_count

    def open_batch_processing_dialog(self) -> None:
        if self.grid_size is None:
            messagebox.showerror("Нет сетки", "Сначала задайте размер ячейки и постройте сетку.")
            return

        zero_count = self._validate_zero_count()
        if zero_count is None:
            return

        folder = filedialog.askdirectory(title="Выберите папку для групповой обработки")
        if not folder:
            return

        root_dir = Path(folder)
        pairs = find_tiff_mask_pairs(root_dir)
        if not pairs:
            messagebox.showinfo("Пары не найдены", "В выбранной папке не найдено ни одной пары TIFF и mask TIFF.")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Обработать группу")
        dialog.geometry("920x560")
        dialog.minsize(760, 420)
        dialog.transient(self.root)
        dialog.grab_set()

        frame = ttk.Frame(dialog, padding=12)
        frame.pack(fill="both", expand=True)

        summary = (
            f"Найдено пар для обработки: {len(pairs)}\n"
            f"Папка: {root_dir}\n"
            f"Размер ячейки: {self.grid_size} px, число нулей N: {zero_count}"
        )
        ttk.Label(frame, text=summary, justify="left").pack(anchor="w")

        ttk.Checkbutton(
            frame,
            text="Сохранить здесь: в текущей директории, в папке processed_arrays",
            variable=self.save_batch_here_var,
        ).pack(anchor="w", pady=(10, 8))

        columns = ("image", "mask", "save")
        tree = ttk.Treeview(frame, columns=columns, show="headings")
        tree.heading("image", text="TIFF")
        tree.heading("mask", text="Mask TIFF")
        tree.heading("save", text="Куда сохранится")
        tree.column("image", width=260, anchor="w")
        tree.column("mask", width=280, anchor="w")
        tree.column("save", width=300, anchor="w")

        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        save_in_processed_dir = self.save_batch_here_var.get()
        for image_path, mask_path in pairs:
            save_path = self._suggest_output_path(
                mask_path=mask_path,
                save_to_processed_dir=save_in_processed_dir,
                batch_root_dir=root_dir,
            )
            tree.insert("", "end", values=(str(image_path), str(mask_path), str(save_path)))

        def refresh_save_targets(*_args) -> None:
            for item in tree.get_children():
                tree.delete(item)
            save_local = self.save_batch_here_var.get()
            for image_path, mask_path in pairs:
                save_path = self._suggest_output_path(
                    mask_path=mask_path,
                    save_to_processed_dir=save_local,
                    batch_root_dir=root_dir,
                )
                tree.insert("", "end", values=(str(image_path), str(mask_path), str(save_path)))

        self.save_batch_here_var.trace_add("write", refresh_save_targets)

        buttons = ttk.Frame(dialog, padding=(12, 0, 12, 12))
        buttons.pack(fill="x")

        def process_pairs() -> None:
            dialog.destroy()
            self.process_batch_pairs(pairs, self.save_batch_here_var.get(), zero_count, root_dir)

        ttk.Button(buttons, text="Запустить обработку", command=process_pairs).pack(side="left")
        ttk.Button(buttons, text="Отмена", command=dialog.destroy).pack(side="right")

    def process_batch_pairs(
        self,
        pairs: list[tuple[Path, Path]],
        save_to_processed_dir: bool,
        zero_count: int,
        batch_root_dir: Path,
    ) -> None:
        processed = 0
        failed: list[str] = []
        for image_path, mask_path in pairs:
            self.status_var.set(f"Обработка пары {processed + 1} из {len(pairs)}: {image_path.name}")
            self.root.update_idletasks()

            image_stack = None
            mask_stack = None
            try:
                image_stack = TiffStack(str(image_path))
                mask_stack = TiffStack(str(mask_path))
                self._validate_stacks(image_stack, mask_stack)
                generated = compute_cell_array(mask_stack, self.grid_size, zero_count)
                output_path = self._suggest_output_path(
                    mask_path=mask_path,
                    save_to_processed_dir=save_to_processed_dir,
                    batch_root_dir=batch_root_dir,
                )
                np.save(output_path, generated)
                processed += 1
            except Exception as exc:
                failed.append(f"{image_path.name}: {exc}")
            finally:
                if image_stack is not None:
                    image_stack.close()
                if mask_stack is not None:
                    mask_stack.close()

        save_mode = (
            f"в {PROCESSED_ARRAYS_DIR}"
            if save_to_processed_dir
            else "рядом с соответствующими mask TIFF"
        )
        if failed:
            self.status_var.set(
                f"Групповая обработка завершена с ошибками. Успешно обработано пар: {processed} из {len(pairs)}."
            )
            messagebox.showwarning(
                "Готово с ошибками",
                "Обработка завершена не для всех пар.\n"
                f"Успешно: {processed} из {len(pairs)}\n"
                f"Сохранение: {save_mode}\n\n"
                f"Ошибки:\n" + "\n".join(failed[:10]),
            )
            return

        self.status_var.set(f"Групповая обработка завершена. Успешно обработано пар: {processed}.")
        messagebox.showinfo("Готово", f"Всё обработалось успешно.\nОбработано пар: {processed}\nСохранение: {save_mode}")

    def build_overlay_image(self, frame_index: int) -> Image.Image | None:
        if self.generated_array is None or self.image_stack is None or self.grid_size is None:
            return None
        values = self.generated_array[frame_index, :, :, 0]
        return create_value_overlay(values, self.image_stack.width, self.image_stack.height, self.grid_size)

    def on_slider_change(self, value: str) -> None:
        if self.image_stack is None:
            return
        self.current_frame = max(0, min(int(float(value)), self.image_stack.frame_count - 1))
        self.render_current_frame()

    def step_frame(self, delta: int) -> None:
        if self.image_stack is None:
            return
        target = max(0, min(self.current_frame + delta, self.image_stack.frame_count - 1))
        if target != self.current_frame:
            self.current_frame = target
            self.frame_slider.set(target)
            self.render_current_frame()

    def render_current_frame(self) -> None:
        if self.image_stack is None or self.mask_stack is None:
            self.frame_label_var.set("Кадр: - / -")
            return

        image_frame = self.image_stack.get_frame(self.current_frame)
        mask_frame = self.mask_stack.get_frame(self.current_frame)

        self.image_panel.show_array_frame(image_frame, self.grid_size)
        self.mask_panel.show_array_frame(mask_frame, self.grid_size)

        overlay = self.build_overlay_image(self.current_frame)
        if overlay is not None:
            self.values_panel.show_pil_image(
                overlay,
                grid_size=self.grid_size,
                image_width=self.image_stack.width,
                image_height=self.image_stack.height,
            )
        else:
            placeholder = Image.new("RGB", (self.image_stack.width, self.image_stack.height), "#202020")
            self.values_panel.show_pil_image(
                placeholder,
                grid_size=self.grid_size,
                image_width=self.image_stack.width,
                image_height=self.image_stack.height,
            )

        self.frame_label_var.set(f"Кадр: {self.current_frame + 1} / {self.image_stack.frame_count}")

    def _close_stacks(self) -> None:
        if self.image_stack is not None:
            self.image_stack.close()
            self.image_stack = None
        if self.mask_stack is not None:
            self.mask_stack.close()
            self.mask_stack = None

    def _on_close(self) -> None:
        self._close_stacks()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    ttk.Style().theme_use("clam")
    TiffGridViewerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
