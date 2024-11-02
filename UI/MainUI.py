import os
from tkinter import *
from tkinter import messagebox, filedialog
from tkinter.ttk import Panedwindow

from Clustering.clustering import cluster_classes, find_k


class MainUI:

    def __init__(self, win):
        wl_window = Panedwindow(win, orient=HORIZONTAL)
        wl_window.place(x=320, y=200)

        # Dataset Window
        dataset_window = Panedwindow(win, orient=HORIZONTAL)
        dataset_window.place(x=50, y=10)
        dataset_frame = LabelFrame(dataset_window, text='Dataset', width=375, height=170)
        dataset_window.add(dataset_frame)

        self.select_button = Button(dataset_frame, text="Select", command=self.select_dataset)
        self.select_button.place(x=10, y=10)

        self.dataset_location = StringVar()
        self.location_entry = Entry(dataset_frame, textvariable=self.dataset_location, state='readonly', width=50)
        self.location_entry.place(x=60, y=15)

        self.is_noisy = BooleanVar()
        self.noise_checkbox = Checkbutton(dataset_frame, text="Add Noise", variable=self.is_noisy)
        self.noise_checkbox.place(x=10, y=50)

        self.chosen_k_label = Label(dataset_frame, text="Number of clusters (K):")
        self.chosen_k_label.place(x=10, y=80)
        self.chosen_k = IntVar()
        self.chosen_k.set(10)
        self.chosen_k_entry = Entry(dataset_frame, textvariable=self.chosen_k, width=9)
        self.chosen_k_entry.place(x=140, y=82)

        self.select_button = Button(dataset_frame, text="Run", command=self.run_clustering, width=12)
        self.select_button.place(x=10, y=120)

        # Optimal K Window
        find_k_window = Panedwindow(dataset_window, orient=HORIZONTAL)
        find_k_window.place(x=220, y=60)
        find_k_frame = LabelFrame(find_k_window, text='Optimal k', width=130, height=100)
        find_k_window.add(find_k_frame)

        self.test_k_label = Label(find_k_frame, text="Test K:")
        self.test_k_label.place(x=10, y=10)
        self.test_k = IntVar()
        self.test_k.set(10)
        self.test_k_entry = Entry(find_k_frame, textvariable=self.test_k, width=9)
        self.test_k_entry.place(x=60, y=12)

        self.optimal_k_button = Button(find_k_frame, text="Find K", command=self.find_optimal_k)
        self.optimal_k_button.place(x=10, y=40)

        # Preprocessing scenarios Window
        preprocessing_window = Panedwindow(win, orient=HORIZONTAL)
        preprocessing_window.place(x=50, y=190)
        preprocessing_frame = LabelFrame(preprocessing_window, text='Preprocessing scenarios', width=265, height=155)
        preprocessing_window.add(preprocessing_frame)

        self.selected_preprocessing = StringVar()
        self.selected_preprocessing.set("WoPP")
        options = ["WoPP", "PPwS", "PPWoS"]
        for option in options:
            self.radio_button = Radiobutton(preprocessing_frame, text=option, variable=self.selected_preprocessing, value=option)
            self.radio_button.pack(anchor=W)

        # Optimal K Window
        concepts_window = Panedwindow(win, orient=HORIZONTAL)
        concepts_window.place(x=200, y=190)
        concepts_frame = LabelFrame(concepts_window, text='Ground truth concepts', width=225, height=80)
        concepts_window.add(concepts_frame)

        self.concepts_label = Label(concepts_frame, text="Concepts (comma separated):")
        self.concepts_label.place(x=8, y=5)
        self.concepts_entry = Entry(concepts_frame, width=25)
        self.concepts_entry.place(x=10, y=30)

    def restart(self):
        python = sys.executable
        os.execl(python, python, *sys.argv)

    # Function to select dataset
    def select_dataset(self):
        folder_path = filedialog.askdirectory()
        if folder_path:  # if a folder is selected
            self.dataset_location.set(folder_path)

    # Function optimal number of clusters
    def find_optimal_k(self):

        dataset_location = self.dataset_location.get()
        prepros_scen = self.selected_preprocessing.get()
        type_dataset = "noisy" if self.is_noisy.get() else "normal"

        if not dataset_location:
            messagebox.showerror("Input Error", "Please select a valid dataset location.")
            return

        self.chosen_k.set(find_k(test_k=self.test_k.get(), solution_path=dataset_location,
                                 type_dataset=type_dataset, type_preprocessing=prepros_scen))

    # Run clustering experiment
    def run_clustering(self):
        dataset_location = self.dataset_location.get()
        prepros_scen = self.selected_preprocessing.get()
        number_k = int(self.chosen_k.get())
        truth_labels = [concept.strip() for concept in self.concepts_entry.get().split(",")]
        type_dataset = "noisy" if self.is_noisy.get() else "normal"

        if not dataset_location:
            messagebox.showerror("Input Error", "Please select a valid dataset location")
            return

        cluster_classes(solution_path=dataset_location, type_dataset=type_dataset,
                        type_preprocessing=prepros_scen, k_clusters=number_k,
                        truth_concepts=truth_labels)


    @staticmethod
    def runUI():
        window = Tk()
        MainUI(window)
        window.title('Source code clustering')
        window.geometry("480x340+10+10")
        icon = PhotoImage(file="UI/logo.png")
        window.iconphoto(True, icon)
        window.mainloop()
