"""Class for the GUI to assist with cluster merging."""

from tkinter import Label
from webcolors import CSS3_HEX_TO_NAMES, hex_to_rgb
from scipy.spatial import KDTree


def css3_lists():
    """Return a list of all css3 color names, and a corresponding list of the colors' RGB values."""
    css3_db = CSS3_HEX_TO_NAMES
    names = []
    rgb_values = []
    for color_hex, color_name in css3_db.items():
        names.append(color_name)
        rgb_values.append(hex_to_rgb(color_hex))

    return names, rgb_values


class ClusterMergerGUI:
    """The GUI to assist with cluster merging."""
    def __init__(self, master, color_list, shape: tuple, image, np_image):
        self.master = master
        master.title("Cluster Merger GUI")

        geometry = "%dx%d+0+0" % (shape[1], shape[0])
        master.geometry(newGeometry=geometry)

        names, rgb_values = css3_lists()
        kdt_db = KDTree(rgb_values)

        self.label = Label(master, image=image)
        self.label.bind('<Button-1>', lambda event: self.leftclick(event, np_image, color_list, names, kdt_db))
        self.label.pack(fill="both", expand=1)

    def leftclick(self, event, np_image, color_list: list, names, kdt_db):
        distance, index = kdt_db.query(np_image[event.y, event.x])
        color = names[index]
        color_list.append(color)
