"""Class for the GUI to assist with cluster merging."""

from tkinter import Label
from webcolors import rgb_to_name


class ClusterMergerGUI:
    """The GUI to assist with cluster merging."""
    def __init__(self, master, color_list, shape, image, np_image):
        self.master = master
        master.title("A simple GUI")

        geometry = "%dx%d+0+0" % (shape[1], shape[0])
        master.geometry(newGeometry=geometry)

        self.label = Label(master, image=image)
        self.label.bind('<Button-1>', lambda event: self.leftclick(event, np_image, color_list))
        self.label.pack(fill="both", expand=1)

    def leftclick(self, event, np_image, color_list):
        color_list.append(rgb_to_name(np_image[event.y, event.x]))
