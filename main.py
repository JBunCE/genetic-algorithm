import cv2
import math
import queue
import random
import warnings
import threading
import statistics
import pandas as pd
import customtkinter as ctk
import matplotlib.pyplot as plt

from typing import List, TypedDict
from sympy import symbols, lambdify
from PIL import Image, ImageTk
from sympy import symbols, lambdify
from matplotlib.animation import FuncAnimation
from pandastable import Table, TableModel
from CTkMessagebox import CTkMessagebox

warnings.filterwarnings(action='ignore')

class Member(TypedDict):
    binary_representation: List[str]
    index: str
    x: float
    fitness: float

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Genetic Algorithm")
        self.geometry(f"{1366}x{768}")

        pd.set_option('display.precision', 10)

        self.options_frame = ctk.CTkFrame(self)
        self.options_frame.pack(side="left", ipadx=10, ipady=10, fill="y", expand=False, padx=10, pady=10)

        self.scroll_frame = ctk.CTkScrollableFrame(self)
        self.scroll_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        self.chart_frame = ctk.CTkFrame(self)
        self.chart_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        self.best_table_frame = ctk.CTkFrame(self.scroll_frame)
        self.best_table_frame.pack(side="top", fill="both", ipadx=10, ipady=10, padx=10, pady=10)

        self.median_table_frame = ctk.CTkFrame(self.scroll_frame)
        self.median_table_frame.pack(side="top", fill="both", ipadx=10, ipady=10, padx=10, pady=10)

        self.worst_table_frame = ctk.CTkFrame(self.scroll_frame)
        self.worst_table_frame.pack(side="top",  fill="both", ipadx=10, ipady=10, padx=10, pady=10)
    
        self.function_label = ctk.CTkLabel(self.options_frame, text="Function:")
        self.function_label.pack(pady=5, padx=10, anchor="w")

        self.function_input = ctk.CTkEntry(self.options_frame, placeholder_text="Enter function here")
        self.function_input.pack(padx=10, anchor="w")

        self.a_interval_label = ctk.CTkLabel(self.options_frame, text="A")
        self.a_interval_label.pack(padx=10, anchor="w")

        self.a_interval_input = ctk.CTkEntry(self.options_frame, placeholder_text="A interval")
        self.a_interval_input.pack(padx=10, pady=5, anchor="w")

        self.b_interval_label = ctk.CTkLabel(self.options_frame, text="B")
        self.b_interval_label.pack(padx=10, anchor="w")

        self.b_interval_input = ctk.CTkEntry(self.options_frame, placeholder_text="B interval")
        self.b_interval_input.pack(padx=10, pady=5, anchor="w")

        self.initial_population_label = ctk.CTkLabel(self.options_frame, text="Initial population:")
        self.initial_population_label.pack(pady=5, padx=10, anchor="w")

        self.initial_population_input = ctk.CTkEntry(self.options_frame, placeholder_text="Enter initial population here")
        self.initial_population_input.pack(padx=10, anchor="w")

        self.population_size_label = ctk.CTkLabel(self.options_frame, text="Population size:")
        self.population_size_label.pack(side="top", pady=5, padx=10, anchor="w")

        self.population_size_input = ctk.CTkEntry(self.options_frame, placeholder_text="Enter population size here")
        self.population_size_input.pack(padx=10, anchor="w")

        self.user_resolution_label = ctk.CTkLabel(self.options_frame, text="User resolution:")
        self.user_resolution_label.pack(pady=5, padx=10, anchor="w")

        self.user_resolution_input = ctk.CTkEntry(self.options_frame, placeholder_text="Enter user resolution here")
        self.user_resolution_input.pack(padx=10, anchor="w")

        self.number_of_generations_label = ctk.CTkLabel(self.options_frame, text="Number of generations:")
        self.number_of_generations_label.pack(pady=5, padx=10, anchor="w")

        self.number_of_generations_input = ctk.CTkEntry(self.options_frame, placeholder_text="Enter number of generations here")
        self.number_of_generations_input.pack(padx=10, anchor="w")

        self.crossover_percent_label = ctk.CTkLabel(self.options_frame, text="Crossover percent")
        self.crossover_percent_label.pack(pady=5, padx=10, anchor="w")

        self.crossover_percent_input = ctk.CTkEntry(self.options_frame, placeholder_text="Enter crossover percent here")
        self.crossover_percent_input.pack(padx=10, anchor="w")

        self.mutation_rate_label = ctk.CTkLabel(self.options_frame, text="Mutation rate [member]:")
        self.mutation_rate_label.pack(pady=5, padx=10, anchor="w")

        self.mutation_rate_input = ctk.CTkEntry(self.options_frame, placeholder_text="Enter mutation rate here")
        self.mutation_rate_input.pack(padx=10, anchor="w")

        self.mutation_rate_gene_label = ctk.CTkLabel(self.options_frame, text="Mutation rate [gene]:")
        self.mutation_rate_gene_label.pack(pady=5, padx=10, anchor="w")

        self.mutation_rate_gene_input = ctk.CTkEntry(self.options_frame, placeholder_text="Enter mutation rate here")
        self.mutation_rate_gene_input.pack(padx=10, anchor="w")

        self.maximization_checkbox = ctk.CTkCheckBox(self.options_frame, text="Maximization")
        self.maximization_checkbox.pack(pady=5, padx=10, anchor="w")

        self.start_button = ctk.CTkButton(self.options_frame, text="Start", command=self.initialization)
        self.start_button.pack(pady=5, padx=10, anchor="w")

        self.prograss_bar = ctk.CTkProgressBar(self.chart_frame, width=200, height=20)
        self.prograss_bar.pack(pady=5, padx=10, anchor="w", fill="x")


        self.prograss_bar.set(0)

        # AG PARAMS

        # USER PARAMS
        self.population: List[Member] = []
        self.maximization = False

        # Solution Space
        self.a = -24
        self.b = 14

        self.initial_population = 10
        self.population_size = 30
        self.user_resolution = 0.076
        self.number_of_generations = 200
        self.crossover_rate = 60
        self.mutation_rate = 50
        self.mutation_rate_gene = 70
        self.percent_of_elitism = 5
        self.breakpoint_line = 0

        # SYSTEM PARAMS
        self.range = 0
        self.points = 0
        self.bits = 0
        self.system_resolution = 0 # Cantidad de decimanles de resolucion de usuario + 1

        # Sympy 
        self.x = symbols("x")
        self.f = None

        # chart
        self.best_x_values: List[Member] = []
        self.best_y_values: List[Member] = []

        self.median_x_values = []
        self.median_y_values = []

        self.worst_x_values: List[Member] = []
        self.worst_y_values: List[Member] = []

        # Dataframes
        self.best_df = pd.DataFrame({"m_index": [0], "x": [0], "fitness": [0], "binary": [""]}).astype({"m_index": int, "x": float, "fitness": str, "binary": str})
        self.median_df = pd.DataFrame({"meadian": [0]}).astype({"meadian": str})
        self.worst_df = pd.DataFrame({"m_index": [0], "x": [0], "fitness": [0], "binary": [""]}).astype({"m_index": int, "x": float, "fitness": str, "binary": str})

        # Table
        self.best_table_df = Table(self.best_table_frame, dataframe=self.best_df, showtoolbar=True, showstatusbar=True)
        self.best_table_df.show()

        self.worst_table = Table(self.worst_table_frame, dataframe=self.worst_df, showtoolbar=True, showstatusbar=True)
        self.worst_table.show()

        self.median_table = Table(self.median_table_frame, dataframe=self.median_df, showtoolbar=True, showstatusbar=True)
        self.median_table.show()

        # Video
        self.canvas = ctk.CTkCanvas(self.chart_frame, bg="black")
        self.canvas.pack(expand=True, fill="both")

        self.queue = queue.Queue()

        self.vide_thread_flag = True
        self.video_thread = threading.Thread(target=self.play_video)
        self.video_thread.start()

    def initialization(self):
        # Reset values
        self.population = []
        self.best_x_values = []
        self.best_y_values = []

        self.median_x_values = []
        self.median_y_values = []

        self.worst_x_values = []
        self.worst_y_values = []

        # Get values from inputs
        try:
            try:
                self.a = float(self.a_interval_input.get())
                self.b = float(self.b_interval_input.get())

                if self.b - self.a < 0:
                    # Lanzar una excepción
                    raise Exception("error")
            except:
                CTkMessagebox(title="Error", message="invalid ranges", icon="cancel")

            self.initial_population = int(self.initial_population_input.get())
            self.population_size = int(self.population_size_input.get())
            self.user_resolution = float(self.user_resolution_input.get())
            self.number_of_generations = int(self.number_of_generations_input.get())
            self.percent_of_elitism = int(self.crossover_percent_input.get())
            self.mutation_rate = int(self.mutation_rate_input.get())
            self.mutation_rate_gene = int(self.mutation_rate_gene_input.get())
            self.maximization = self.maximization_checkbox.get()
        except:
            CTkMessagebox(title="Error", message="Error on some input", icon="cancel")

        # Calculate Params
        self.range = self.b - self.a
        self.points = (self.range / self.user_resolution) + 1
        self.bits = math.ceil(math.log2(self.points))
        self.system_resolution = self.range / ((2 ** self.bits) - 1)

        # Lambdyfy the function
        self.f = lambdify(self.x, self.function_input.get())
        # self.f = lambdify(self.x, "x * cos(x)")

        # Generate initial population
        for _ in range(self.initial_population):
            member = Member()
            member["index"] = random.randint(1, (2 ** self.bits))
            member["binary_representation"] = list(format(member["index"], '0{}b'.format(self.bits)))
            member["x"] = self.a + member["index"] * self.system_resolution
            member["fitness"] = 0
            self.population.append(member)
        
        self.breakpoint_line = random.randint(1, len(self.population[0]["binary_representation"]))

        # Member evaluation
        for member in self.population:
            if member["fitness"] == 0:
                member["fitness"] = self.f(member["x"])
        
        self.sort()
        self._optimization()

    def _optimization(self):
        for geneartion in range(self.number_of_generations):
            self._crossover()
            self._mutation()
            self._evaluation()
            
            self.sort()
            # Eliminar repetidos
            seen = set()
            self.population = [member for member in self.population if member["index"] not in seen and not seen.add(member["index"])]
            self.sort()

            fitness = [member["fitness"] for member in self.population]

            self.make_chart(geneartion, mean=statistics.median(fitness))
            self._pode(mean=statistics.median(fitness))
            print(f"Generation: {geneartion}")

        video_thread = threading.Thread(target=self.make_video)
        video_thread.start()

    def _evaluation(self):
        for member in self.population:
            if member["fitness"] == 0:
                member["index"] = int("".join(member["binary_representation"]), 2)
                member["x"] = self.a + member["index"] * self.system_resolution
                member["fitness"] = self.f(member["x"])

    def _crossover(self):
        # Make pairs
        percent = ((self.percent_of_elitism * len(self.population)) // 100)
        best_members = self.population[:percent]
        selected_members_to_crossover = self.population[percent:]

        for selected_member in selected_members_to_crossover:
            for best_member in best_members:
                first_child = Member()
                second_child = Member()

                first_child["index"] = None
                second_child["index"] = None
                first_child["fitness"] = 0
                second_child["fitness"] = 0

                first_child["binary_representation"] = selected_member["binary_representation"][:self.breakpoint_line] + best_member["binary_representation"][self.breakpoint_line:]
                second_child["binary_representation"] = selected_member["binary_representation"][self.breakpoint_line:] + best_member["binary_representation"][:self.breakpoint_line]

                self.population.append(first_child)
                self.population.append(second_child)

    def _mutation(self):
        for member in self.population:
            if member["index"] == None:
                if random.randint(1, 100) <= self.mutation_rate:
                    for i in range(len(member["binary_representation"])):
                        if random.randint(1, 100) <= self.mutation_rate_gene:
                            member["binary_representation"][i] = str(random.randint(0, 1))

    def _pode(self, mean):
        print(mean)
        # Indice media [1, 2, 3] [3, 2, 1]
        if self.maximization:
            index_mean = next(i for i, d in reversed(list(enumerate(self.population))) if d["fitness"] >= mean)
        else:
            index_mean = next(i for i, d in reversed(list(enumerate(self.population))) if d["fitness"] <= mean)

        # Mantener tamanio de poblacion
        self.population = self.population[:index_mean]
        self.population = self.population[:self.population_size]
        
    def sort(self):
        self.population = sorted(self.population, key=lambda x: x["fitness"], reverse=self.maximization)

    def make_chart(self, generation, mean):
        self.best_x_values.append(generation)
        self.best_y_values.append(self.population[0])

        df = pd.DataFrame({
            "m_index": [v["index"] for v in self.best_y_values],
            "x": [v["x"] for v in self.best_y_values],
            "fitness": [str(v["fitness"]) for v in self.best_y_values],
            "binary": ["".join(v["binary_representation"]) for v in self.best_y_values]
        })
        self.best_table_df.updateModel(TableModel(df))
        self.best_table_df.redraw()

        self.median_x_values.append(generation)
        self.median_y_values.append(mean)

        df = pd.DataFrame({
            "meadian": [str(m) for m in self.median_y_values]
        })
        self.median_table.updateModel(TableModel(df))
        self.median_table.redraw()

        self.worst_x_values.append(generation)
        self.worst_y_values.append(self.population[-1])

        df = pd.DataFrame({
            "m_index": [v["index"] for v in self.worst_y_values],
            "x": [v["x"] for v in self.worst_y_values],
            "fitness": [str(v["fitness"]) for v in self.worst_y_values],
            "binary": ["".join(v["binary_representation"]) for v in self.worst_y_values]
        })
        self.worst_table.updateModel(TableModel(df))
        self.worst_table.redraw()

        self.prograss_bar.set((generation / self.number_of_generations) * 100)

    def make_video(self):
        fig = plt.figure(figsize=(9.5, 9))
        self.ax = fig.add_subplot(111)

        animation = FuncAnimation(fig, self.update, frames=len(self.best_x_values), repeat=False)
        animation.save("final.mp4", writer="ffmpeg")
        self.queue.put("reset")
    
    def update(self, frame):
        self.ax.clear()

        self.ax.title.set_text("Population evolution")
        self.ax.set_xlabel("Generations")
        self.ax.set_ylabel("Fitness")

        self.ax.set_xlim([0,self.number_of_generations])

        self.ax.plot(self.best_x_values[:frame], [d["fitness"] for d in self.best_y_values[:frame]], label="Best", color="blue", linestyle="-")
        self.ax.plot(self.median_x_values[:frame], self.median_y_values[:frame], label="Median", color="green", linestyle="-")
        self.ax.plot(self.worst_x_values[:frame], [d["fitness"] for d in self.worst_y_values[:frame]], label="Worst", color="red", linestyle="-")
        self.ax.legend(labels=["Best", "Median", "Worst"])

    def play_video(self):
        while self.vide_thread_flag:
            command = self.queue.get()
            if command == "reset":
                self.cap = cv2.VideoCapture("final.mp4")
                self.update_image()

    def update_image(self):
        ret, frame = self.cap.read()

        if ret:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)
            self.canvas.create_image(0, 0, anchor='nw', image=image)

            self.canvas.image = image

        self.after(100, self.update_image)

    def reset_video(self):
        self.video_index += 1
        if self.video_index < len(self.functions_entries):
            self.video_index = 0
        self.queue.put("reset")

if __name__ == "__main__":
    app = App()
    app.mainloop()