import glob
import tkinter
import customtkinter as ctk
from pathlib import Path
from functools import partial
from deepmriprep.ui.cli import get_paths_from_csv
from deepmriprep.preprocess import IO, OUTPUTS, DIR_FORMATS, run_preprocess, find_bids_t1w_files
from deepmriprep.utils import DATA_PATH
OUTPUT_MODES = list(OUTPUTS)[:3] + ['custom']
STEP_LABELS = {'bet': 'Brain Extraction', 'affine': 'Affine Registration', 'segment_brain': 'Tissue Segmentation',
               'segment_nogm': 'Tissue Probabilities', 'warp': 'Nonlinear Registration', 'smooth': 'Smoothing',
               'atlas': 'Atlases'}


class App(ctk.CTk):
    def __init__(self, title='deepmriprep', padx=2, pady=2, sticky='nsew'):
        super().__init__()

        self.title(title)
        self.grid_kwargs = {'padx': padx, 'pady': pady, 'sticky': sticky}

        self.input_files = []
        self.bids_dir = None
        self.output_paths = None
        self.dir_format = DIR_FORMATS[0]
        self.outputs = OUTPUT_MODES[0]
        self.no_gpu = False
        self.custom_outputs = {o: False for o in OUTPUTS['all']}
        self.custom_step_labels = {}
        self.custom_checkboxes = {}

        self.bids_dir_button = ctk.CTkButton(self, text='Select BIDS Folder',
                                             command=partial(self.set_input, mode='bids'))
        self.bids_dir_button.grid(row=0, column=0, **self.grid_kwargs)
        self.input_file_button = ctk.CTkButton(self, text='Select Input Files',
                                               command=partial(self.set_input, mode='files'))
        self.input_file_button.grid(row=0, column=1, **self.grid_kwargs)
        placeholder_text = 'Filepattern e.g. /path/to/input/*T1w.nii.gz'
        self.input_pattern_entry = ctk.CTkEntry(self, placeholder_text=placeholder_text)
        self.input_pattern_entry.bind('<Return>', command=self.set_input)
        self.input_pattern_entry.grid(row=0, column=2, columnspan=3, **self.grid_kwargs)
        self.path_csv_button = ctk.CTkButton(self, text='Select CSV with Filepaths',
                                             command=partial(self.set_input, mode='csv'))
        self.path_csv_button.grid(row=1, column=0, **self.grid_kwargs)
        self.output_dir_button = ctk.CTkButton(self, state='disabled', text='Select Output Folder',
                                               command=self.set_output_dir)
        self.output_dir_button.grid(row=1, column=1, **self.grid_kwargs)
        self.output_dir_entry = ctk.CTkEntry(self, state='disabled')
        self.output_dir_entry.bind('<Return>', command=partial(self.set_output_dir, filedialog=False))
        self.output_dir_entry.grid(row=1, column=2, columnspan=3, **self.grid_kwargs)
        self.dir_format_label = ctk.CTkLabel(self, text='Directory Format:')
        self.dir_format_label.grid(row=2, column=0, **self.grid_kwargs)
        self.dir_format_dropdown = ctk.CTkOptionMenu(self, values=DIR_FORMATS, command=self.set_dir_format,
                                                   state='disabled')
        self.dir_format_dropdown.grid(row=2, column=1, **self.grid_kwargs)
        self.outputs_label = ctk.CTkLabel(self, text='Output Modalities:')
        self.outputs_label.grid(row=2, column=2, columnspan=2, **self.grid_kwargs)
        self.outputs_dropdown = ctk.CTkOptionMenu(self, state='disabled', values=OUTPUT_MODES, command=self.set_outputs)
        self.outputs_dropdown.grid(row=2, column=4, **self.grid_kwargs)
        self.status_label = ctk.CTkLabel(self, text='Selected 0 Input Files', text_color='gray')
        self.status_label.grid(row=3, column=0, **self.grid_kwargs)
        self.run_button = ctk.CTkButton(self, state='disabled', text='Run', command=self.run_processing)
        self.run_button.grid(row=3, column=1, columnspan=3, **self.grid_kwargs)
        self.no_gpu_checkbox = ctk.CTkCheckBox(self, state='disabled', text='No GPU', command=self.set_no_gpu)
        self.no_gpu_checkbox.grid(row=3, column=4, **self.grid_kwargs)
        self.progress_bar = ctk.CTkProgressBar(self)
        self.progress_bar.set(0)
        self.progress_bar.grid(row=4, column=0, columnspan=5, **self.grid_kwargs)

    def set_input(self, event=None, mode='pattern'):
        if mode == 'bids':
            bids_dir = ctk.filedialog.askdirectory()
            files = find_bids_t1w_files(bids_dir) if bids_dir else []
            if bids_dir:
                self.bids_dir = bids_dir
        elif mode == 'csv':
            csv_path = ctk.filedialog.askopenfilename(filetypes=[('CSV File', '*.csv')])
            if csv_path.endswith('.csv'):
                table_dir = str(Path(csv_path).parent)
                self.output_dir_entry.configure(state='normal')
                self.output_dir_entry.insert(0, table_dir)
                self.output_dir_entry.configure(state='disabled')
                files, self.output_paths = get_paths_from_csv(csv_path)
            else:
                files = []
        elif mode == 'pattern':
            pattern = self.input_pattern_entry.get()
            files = [fp for fp in sorted(glob.glob(pattern)) if fp.endswith(('.nii.gz', '.nii'))]
            if len(files) == 0:
                print(f'Found no files based on given pattern {pattern}')
        else:
            files = ctk.filedialog.askopenfilenames(filetypes=[('Image Files', '*.nii *.nii.gz')])
        if files:
            self.input_files = files
            self.status_label.configure(text=f'Selected {len(files)} Input Files')
            for item in [self.bids_dir_button, self.input_file_button, self.input_pattern_entry, self.path_csv_button]:
                item.configure(state='disabled')
            if mode == 'bids':
                self.output_dir_entry.insert(0, f'{self.bids_dir}/derivatives')
                enable_items = [self.dir_format_dropdown, self.outputs_dropdown, self.no_gpu_checkbox, self.run_button]
            elif mode == 'csv':
                enable_items = [self.no_gpu_checkbox, self.run_button]
            else:
                enable_items = [self.output_dir_button, self.output_dir_entry]
            for item in enable_items:
                item.configure(state='normal')
            if mode not in ['csv', 'bids']:
                self.output_dir_entry.configure(placeholder_text='e.g. /path/to/output/directory')

    def set_output_dir(self, event=None, filedialog=True):
        folder = ctk.filedialog.askdirectory() if filedialog else self.output_dir_entry.get()
        if folder:
            self.output_dir_entry.insert(0, folder)
            self.output_dir_entry.delete(len(folder), ctk.END)
            self.output_dir_entry.configure(state='disabled')
            for item in [self.dir_format_dropdown, self.outputs_dropdown, self.no_gpu_checkbox, self.run_button]:
                item.configure(state='normal')
            self.output_dir_button.configure(state='disabled')

    def set_dir_format(self, dir_format):
        self.dir_format = dir_format

    def set_outputs(self, *args):
        columns = 4
        if self.outputs_dropdown.get() == 'custom' and self.outputs != 'custom':
            row = 5
            for i, (step, io_dict) in enumerate(IO.items()):
                label = ctk.CTkLabel(self, text=STEP_LABELS[step] + ':')
                self.custom_step_labels.update({step: label})
                label.grid(row=row, column=0, **self.grid_kwargs)
                for j, output in enumerate(io_dict['output']):
                    if (j % columns) + 1 == 1 and j > 0:
                        row += 1
                    checkbox = ctk.CTkCheckBox(self, text=output[:16], command=partial(self.set_checkbox, output))
                    checkbox.grid(row=row, column=(j % columns) + 1, **self.grid_kwargs)
                    self.custom_checkboxes.update({output: checkbox})
                row += 1
        elif self.outputs_dropdown.get() != 'custom' and self.outputs == 'custom':
            for checkbox in self.custom_checkboxes.values():
                checkbox.grid_remove()
            for label in self.custom_step_labels.values():
                label.grid_remove()
        self.outputs = self.outputs_dropdown.get()

    def set_no_gpu(self):
        self.no_gpu = self.no_gpu_checkbox.get()

    def set_checkbox(self, output):
        self.custom_outputs[output] = self.custom_checkboxes[output].get() == 1

    def update_progress(self, progress_bar):
        self.progress_bar.set(progress_bar.n / progress_bar.total)
        if progress_bar.n > 0:
            self.status_label.configure(text=progress_bar.__str__().split('| ')[-1].replace('it', '🧠'))
        self.update_idletasks()

    def run_processing(self):
        self.status_label.configure(text='Warming up...')
        self.update_idletasks()
        input_paths = self.input_files if self.bids_dir is None else None
        if self.outputs == 'custom':
            outputs = [k for k, v in self.custom_outputs.items() if v]
        else:
            outputs = OUTPUTS[self.outputs]
        if len(outputs) == 0:
            print('No output modality selected. To enable processing, please select at least one output modality!')
            return None
        run_preprocess(input_paths, self.bids_dir, self.output_paths, self.output_dir_entry.get(), outputs,
                       self.dir_format, self.no_gpu, self.update_progress)
        self.destroy()


def run_gui():
    app = App()
    icon = tkinter.PhotoImage(file=f'{DATA_PATH}/icon.png')
    app.iconphoto(True, icon)
    app.mainloop()


if __name__ == '__main__':
    run_gui()
