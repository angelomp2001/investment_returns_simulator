import ttkbootstrap as tb
import tkinter as tk
from get_data_chatgpt import get_symbol_data
import pandas as pd

class Dashboard:
    
    def __init__(self, theme = "darkly"):
        # Create a window with the specified theme
        self.root = tb.Window(themename=theme)
        self.root.title("Investment Returns Simulator")
        self.root.geometry("955x1000+955+000")  # widthxheight+pos_x+pos_y
        self._build_gui()
        self.root.mainloop()
        self.df

    def _build_gui(self):
        self.nb = tb.Notebook(self.root); self.nb.pack(fill="both", expand=True)
        df = get_symbol_data(symbols=["VOOG"])
        self.df = df

        self._overview_tab()
        # self._analysis_tab()
        # self._data_tab()

        # ─────────────────  OVERVIEW TAB  ─────────────────────────
    def _overview_tab(self):
        tab = tb.Frame(self.nb); self.nb.add(tab, text="Charts")
        self._filters(tab)
        
    def _spin(self, parent, **kw):
        return tb.Spinbox(parent, style="White.TSpinbox", **kw)

    def _filters(self, parent):
        f = tb.Labelframe(parent, text="Filters", padding=6)
        f.pack(fill="x", padx=8, pady=6)
        tk.Label(f, text="Make").grid(row=0, column=0, sticky="w", padx=4)
        self.make = tk.StringVar(value="All")
        self.drive = tk.StringVar(value="All")
        tb.Combobox(f, textvariable=self.make, state="readonly", width=14,
                    values=["All"] + sorted(self.df['VOOG'].unique()),
                    bootstyle="dark")\
          .grid(row=0, column=1)
        
        pr_min, pr_max = self.df["VOOG"].min(), self.df["VOOG"].max()
        tk.Label(f, text="Price $").grid(row=0, column=4, padx=(20, 4))
        self.pmin = tk.DoubleVar(value=float(pr_min))
        self.pmax = tk.DoubleVar(value=float(pr_max))
        for col, var in [(5, self.pmin), (6, self.pmax)]:
            self._spin(f, from_=0, to=float(pr_max), textvariable=var,
                       width=10, increment=1000, bootstyle="secondary")\
              .grid(row=0, column=col)
        
        tb.Button(f, text="Apply Year/Price Filters",
                  bootstyle="primary-outline",
                  command=self._apply_filters)\
          .grid(row=0, column=10, padx=(30, 4))
        
    # ───────────────── FILTER & STATS ──────────────────────────
    def _apply_filters(self, *_):
        df = self.df.copy()
        if self.make.get() != "All":
            df = df[df["VOOG"] == self.make.get()]
        if hasattr(self, "drive") and self.drive.get() != "All":
            df = df[df["drivetrain"] == self.drive.get()]
        try:
            pmin, pmax = float(self.pmin.get()), float(self.pmax.get())
        except ValueError:
            pmin, pmax = df["price"].min(), df["price"].max()
        df = df[(df["price"] >= pmin) & (df["price"] <= pmax)]
        if "year" in df.columns and hasattr(self, "ymin"):
            try:
                ymin, ymax = int(self.ymin.get()), int(self.ymax.get())
            except ValueError:
                ymin, ymax = df["year"].min(), df["year"].max()
            df = df[(df["year"] >= ymin) & (df["year"] <= ymax)]
        self.filtered = df
        self._update_cards()
        self._draw_overview()
        self._fill_tree()
        if self.current_analysis_plot_func:
            self.current_analysis_plot_func()



Dashboard()
