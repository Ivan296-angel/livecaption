"""Step 31 UI refresh: floating, borderless Tkinter window for Step 30.8.

No changes to caption event contract, GPU scheduling or checkpoint logic.
Keyboard shortcuts work when the floating Tkinter window has focus.
"""
import queue
import time
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

from caption_freshness import audio_clock

FRESHNESS_WARNING_SECONDS = 1.5
WINDOW_ALPHA = 0.94  # 0.0 fully transparent, 1.0 opaque; Windows supported.
BG = "#111827"
PANEL = "#1F2937"
BORDER = "#374151"
FG = "#F9FAFB"
MUTED = "#9CA3AF"


class CaptionUI:
    def __init__(self, on_close=None):
        self.root = tk.Tk()
        self.root.title("Live English Captions")
        self.root.geometry("860x540+120+100")
        self.root.minsize(550, 350)
        self.root.configure(bg=BG)
        self.root.overrideredirect(True)  # Remove native window border/title bar.
        try:
            self.root.attributes("-alpha", WINDOW_ALPHA)
        except tk.TclError:
            pass  # Transparency unsupported by some platforms/window managers.
        try:
            self.root.attributes("-topmost", True)
        except tk.TclError:
            pass

        self.events = queue.Queue()
        self.on_close = on_close
        self.closing = False
        self.active_phrase = None
        self.last_live_endpoint = -1
        self.last_live_displayed_at = None
        self.current_live_text = ""
        self.live_boundary = 0
        self.frozen_live = {}  # checkpoint end_sample -> frozen provisional text
        self.drag_origin = None
        self._poll_id = None

        frame = tk.Frame(self.root, bg=BG, highlightthickness=1,
                         highlightbackground=BORDER)
        frame.pack(fill="both", expand=True)

        # Only the title region is draggable, keeping text selection functional.
        header = tk.Frame(frame, bg=BG, height=52, cursor="fleur")
        header.pack(fill="x", padx=12, pady=(9, 3))
        header.pack_propagate(False)
        title = tk.Label(header, text="LIVE ENGLISH CAPTIONS", bg=BG,
                         fg=FG, font=("Segoe UI", 12, "bold"), cursor="fleur")
        title.pack(side="left", padx=(8, 10))
        for widget in (header, title):
            widget.bind("<ButtonPress-1>", self._drag_start)
            widget.bind("<B1-Motion>", self._drag_move)
            widget.bind("<ButtonRelease-1>", self._drag_end)

        tk.Button(header, text="✕", command=self.request_close,
                  bg="#7F1D1D", fg="white", relief="flat", bd=0,
                  font=("Segoe UI", 12, "bold"), width=3, cursor="hand2",
                  activebackground="#991B1B", activeforeground="white").pack(side="right", padx=(7, 3))
        tk.Button(header, text="Save  Ctrl+S", command=self.save_final_transcript,
                  bg="#334155", fg="white", relief="flat", bd=0,
                  font=("Segoe UI", 9), padx=10, pady=5, cursor="hand2",
                  activebackground="#475569", activeforeground="white").pack(side="right", padx=3)

        self.status_var = tk.StringVar(value="Starting...")
        tk.Label(frame, textvariable=self.status_var, font=("Segoe UI", 9),
                 fg=MUTED, bg=BG).pack(anchor="w", padx=20, pady=(0, 8))

        tk.Label(frame, text="FINAL TRANSCRIPT  ·  Select text to copy",
                 font=("Segoe UI", 10, "bold"), fg="#34D399", bg=BG,
                 ).pack(anchor="w", padx=20)
        self.final_text = scrolledtext.ScrolledText(
            frame, wrap=tk.WORD, font=("Segoe UI", 12),
            bg=PANEL, fg=FG, selectbackground="#2563EB", selectforeground="white",
            insertbackground="white", height=11, state="disabled", bd=0,
            padx=10, pady=8, relief="flat", takefocus=True,
        )
        self.final_text.pack(fill="both", expand=True, padx=20, pady=(8, 12))

        tk.Label(frame, text="LIVE CAPTION  ·  Provisional / selectable",
                 font=("Segoe UI", 10, "bold"), fg="#60A5FA", bg=BG,
                 ).pack(anchor="w", padx=20)
        self.live_text = scrolledtext.ScrolledText(
            frame, wrap=tk.WORD, font=("Segoe UI", 13),
            bg=PANEL, fg="#BFDBFE", selectbackground="#2563EB", selectforeground="white",
            insertbackground="#BFDBFE", height=6, state="disabled", bd=0,
            padx=10, pady=8, relief="flat", takefocus=True,
        )
        self.live_text.pack(fill="x", padx=20, pady=(8, 8))

        footer = tk.Label(frame, text="Drag title to move  •  Ctrl+S save FINAL  •  Ctrl+T close  •  Ctrl+C copy selection",
                          bg=BG, fg=MUTED, font=("Segoe UI", 8))
        footer.pack(anchor="w", padx=20, pady=(0, 11))

        self._install_copy_menu(self.final_text)
        self._install_copy_menu(self.live_text)
        self.root.bind_all("<Control-s>", self._save_shortcut)
        self.root.bind_all("<Control-t>", self._close_shortcut)
        self.root.bind_all("<Control-c>", self._copy_shortcut)
        self.root.bind_all("<Control-S>", self._save_shortcut)
        self.root.bind_all("<Control-T>", self._close_shortcut)
        self.root.bind_all("<Control-C>", self._copy_shortcut)
        self.root.protocol("WM_DELETE_WINDOW", self.request_close)

        self.set_live("Waiting for speech...")
        self._poll_id = self.root.after(50, self.process_events)

    def _drag_start(self, event):
        self.drag_origin = (event.x_root, event.y_root,
                            self.root.winfo_x(), self.root.winfo_y())

    def _drag_move(self, event):
        if self.drag_origin is None:
            return
        mx, my, wx, wy = self.drag_origin
        x = wx + event.x_root - mx
        y = wy + event.y_root - my
        self.root.geometry(f"+{x}+{y}")

    def _drag_end(self, _event):
        self.drag_origin = None

    def _install_copy_menu(self, widget):
        menu = tk.Menu(widget, tearoff=0)
        menu.add_command(label="Copy", command=lambda w=widget: self.copy_selection(w))
        menu.add_command(label="Select All", command=lambda w=widget: self.select_all(w))

        def show(event):
            try:
                widget.focus_set()
                menu.tk_popup(event.x_root, event.y_root)
            finally:
                menu.grab_release()
        widget.bind("<Button-3>", show)

    @staticmethod
    def select_all(widget):
        widget.tag_add(tk.SEL, "1.0", "end-1c")
        widget.mark_set(tk.INSERT, "1.0")
        widget.see("1.0")

    def copy_selection(self, widget):
        try:
            selected = widget.get(tk.SEL_FIRST, tk.SEL_LAST)
        except tk.TclError:
            return False
        if not selected:
            return False
        self.root.clipboard_clear()
        self.root.clipboard_append(selected)
        self.root.update_idletasks()
        return True

    def _copy_shortcut(self, event):
        widget = event.widget
        if widget in (self.final_text, self.live_text):
            self.copy_selection(widget)
            return "break"
        # Do not intercept Ctrl+C in any future editable form fields.

    def _save_shortcut(self, _event):
        self.save_final_transcript()
        return "break"

    def _close_shortcut(self, _event):
        self.request_close()
        return "break"

    def save_final_transcript(self):
        """Save only FINAL text, not provisional LIVE hypotheses."""
        if self.closing:
            return
        content = self.final_text.get("1.0", "end-1c").strip()
        if not content:
            messagebox.showinfo("No final transcript", "There is no final transcript to save yet.", parent=self.root)
            return
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save final transcript",
            defaultextension=".txt",
            initialfile="final_transcript.txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(content + "\n")
        except OSError as exc:
            messagebox.showerror("Save failed", str(exc), parent=self.root)
        else:
            self.status_var.set(f"Saved final transcript: {path}")

    def send(self, kind, phrase_id=None, text="", timing=None):
        """Thread-safe entry point from the audio/GPU workers."""
        self.events.put((kind, phrase_id, text, timing))

    def add_final(self, text):
        """Tk main thread only. Preserve user's selection and scroll position when possible."""
        if not text:
            return
        widget = self.final_text
        at_bottom = widget.yview()[1] >= 0.98
        widget.configure(state="normal")
        widget.insert(tk.END, text + "\n\n")
        widget.configure(state="disabled")
        if at_bottom:
            widget.see(tk.END)

    def set_live(self, text):
        """Tk main thread only. Don't overwrite LIVE text while selecting it."""
        widget = self.live_text
        try:
            selected = widget.tag_ranges(tk.SEL)
        except tk.TclError:
            selected = ()
        if selected:
            # The hypothesis continues to update internally; refresh after selection clears.
            return
        widget.configure(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, text)
        widget.configure(state="disabled")
        widget.see(tk.END)

    def refresh_live(self):
        parts = [value for value in self.frozen_live.values() if value]
        if self.current_live_text:
            parts.append(self.current_live_text)
        self.set_live("\n\n".join(parts) if parts else "Listening...")

    def process_events(self):
        if self.closing and self.on_close is None:
            return
        try:
            while True:
                kind, phrase_id, text, timing = self.events.get_nowait()
                if kind == "status":
                    self.status_var.set(text)
                elif kind == "start":
                    self.active_phrase = phrase_id
                    self.last_live_endpoint = -1
                    self.last_live_displayed_at = None
                    self.current_live_text = ""
                    self.live_boundary = 0
                    self.frozen_live.clear()
                    self.status_var.set("Speech detected")
                    self.set_live("Listening...")
                elif kind == "split":
                    if phrase_id == self.active_phrase and timing is not None:
                        boundary = timing["end_sample"]
                        if boundary > self.live_boundary:
                            self.frozen_live[boundary] = self.current_live_text
                            self.current_live_text = ""
                            self.live_boundary = boundary
                            self.refresh_live()
                elif kind == "live":
                    if (phrase_id == self.active_phrase
                            and timing is not None
                            and timing["live_start_sample"] >= self.live_boundary
                            and timing["endpoint_frame"] > self.last_live_endpoint):
                        self.current_live_text = text
                        self.refresh_live()
                        displayed_at = time.perf_counter()
                        capture_to_submit = max(0, timing["submitted_at"] - timing["captured_at"])
                        queue_delay = max(0, timing["started_at"] - timing["submitted_at"])
                        gpu_delay = max(0, timing["finished_at"] - timing["started_at"])
                        ui_delay = max(0, displayed_at - timing["finished_at"])
                        capture_to_display = max(0, displayed_at - timing["captured_at"])
                        latest_audio = audio_clock.snapshot()
                        endpoint_frame = timing["endpoint_frame"]
                        sample_rate = timing["sample_rate"]
                        audio_behind_seconds = max(0.0, (latest_audio.frames - endpoint_frame) / sample_rate)
                        endpoint_age = max(0.0, displayed_at - timing["captured_at"])
                        is_out_of_order = endpoint_frame <= self.last_live_endpoint
                        stale = audio_behind_seconds > FRESHNESS_WARNING_SECONDS or is_out_of_order
                        display_gap = (None if self.last_live_displayed_at is None
                                       else displayed_at - self.last_live_displayed_at)
                        self.last_live_endpoint = max(self.last_live_endpoint, endpoint_frame)
                        self.last_live_displayed_at = displayed_at
                        print(f"[LIVE E2E #{phrase_id}] "
                              f"Capture->Submit={capture_to_submit:.3f}s | "
                              f"Queue={queue_delay:.3f}s | GPU={gpu_delay:.3f}s | "
                              f"UI={ui_delay:.3f}s | Capture->Display={capture_to_display:.3f}s | "
                              f"Audio={timing['audio_seconds']:.1f}s")
                        gap_label = "first" if display_gap is None else f"{display_gap:.3f}s"
                        print(f"[LIVE FRESH #{phrase_id}] "
                              f"EndpointAge={endpoint_age:.3f}s | AudioBehind={audio_behind_seconds:.3f}s | "
                              f"DisplayGap={gap_label} | OutOfOrder={'YES' if is_out_of_order else 'NO'} | "
                              f"Stale={'YES' if stale else 'NO'}")
                elif kind == "end":
                    if phrase_id == self.active_phrase:
                        self.status_var.set("Finalizing phrase...")
                elif kind == "checkpoint":
                    if text:
                        self.add_final(text)
                    if phrase_id == self.active_phrase and timing is not None:
                        end_sample = timing["end_sample"]
                        if end_sample is not None:
                            self.frozen_live = {
                                end: value for end, value in self.frozen_live.items()
                                if end > end_sample
                            }
                            self.refresh_live()
                elif kind == "final":
                    if text:
                        self.add_final(text)
                    if phrase_id == self.active_phrase:
                        self.active_phrase = None
                        self.frozen_live.clear()
                        self.current_live_text = ""
                        self.live_boundary = 0
                        self.set_live("Waiting for speech...")
                        self.status_var.set("Listening...")
                elif kind == "error":
                    self.status_var.set(text)
                elif kind == "stopped":
                    self.status_var.set("Stopped")
                    if self.closing:
                        self.root.destroy()
                        return
        except queue.Empty:
            pass
        self._poll_id = self.root.after(50, self.process_events)

    def request_close(self):
        if self.closing:
            return
        self.closing = True
        self.status_var.set("Stopping...")
        self.set_live("Finishing pending captions...")
        if self.on_close is not None:
            self.on_close()
        else:
            self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    ui = CaptionUI()
    ui.add_final("Hi, I'm Catherine. And I'm Raff. We're the founders of Colonel.")
    ui.set_live("Colonel is a developer platform providing infrastructure...")
    ui.run()
