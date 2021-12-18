#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import drawSvg as draw
import tempfile

class objdict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)

drivers_colors_list = [
    '#faff18',
    '#a818ff',
    '#3318ff',
    '#18aeff',
    '#18ffce',
    '#18ff59',
    '#78ff18',
    '#ff7a18',
    '#ff18dd',
    '#ff1818',
    '#ff186c',
    '#25731b',
    '#73731b',
    '#1b7367',
    '#1b4a73',
    '#4f1b73',
    '#1b7351',
]

colors=objdict({
    'white' : 'f5f5dc',
    'red': 'red',
    'blue': 'blue',
    'green': 'green',
    'dark_red': '#8b5962',
    'dark_blue': '#59598b',
    'dark_green': '#516e4a',
    'black': 'black',
    'grey': '#555455',
    'orange': '#ff4500',
    'beige': '#ffebcd',
    'dark_beige': '#321414',
})

class Hyperlink(draw.DrawingParentElement):
    TAG_NAME = 'a'
    def __init__(self, **kwargs):
        # Other init logic...
        # Keyword arguments to super().__init__() correspond to SVG node
        # arguments: stroke_width=5 -> stroke-width="5"
        super().__init__(**kwargs)


# Draw the mini car
def mini_car(drawing, position, size, identity):
    drawing.append(draw.Rectangle(position[0]/size, position[1]/size, 0.05, 0.04,
            fill=drivers_colors_list[identity - 1], stroke_width=0.01, stroke=colors.black))
    drawing.append(draw.Circle(position[0]/size, position[1]/size, 0.02,
        fill=drivers_colors_list[identity - 1], stroke_width=0.01, stroke=colors.black))
    drawing.append(draw.Circle(position[0]/size + 0.05, position[1]/size, 0.02,
        fill=drivers_colors_list[identity - 1], stroke_width=0.01, stroke=colors.black))


arrow = draw.Marker(-0.1, -0.5, 0.9, 0.5, scale=4, orient='auto')
arrow.append(draw.Lines(-0.1, -0.5, -0.1, 0.5, 0.9, 0, fill='red', close=True))


def instance2Image_rep(targets, drivers, size, time_step, time_end, out='svg'):
    # Return an image gathered from svg data
    default_size = 300
    time_margin = 0.01
    time_hight = 0.05 * len(targets)

    final = draw.Drawing(2.5, 2+time_hight, origin=(0,0))

    data_panel = draw.Group(id='data', origin=(0,0), transform='translate(2, 0)') #draw.Drawing(0.5, 2, origin=(0,0))
    data_panel.append(draw.Rectangle(0, 0, 0.5, 2, fill=colors.beige))

    time_table = draw.Group(id='time', origin=(0,0), transform='translate(0, -2)') #draw.Drawing(2, time_hight, origin=(0,0), fill=colors.white)
    time_table.append(draw.Rectangle(0, 0, 2, time_hight, fill=colors.beige))

    corner_info = draw.Group(id='info', origin=(0,0), transform='translate(2, -2)')
    corner_info.append(draw.Rectangle(0, 0, 0.5, time_hight, fill=colors.beige))
    text_lines = ['- DARP Simulation -', 'Time: ' + str(time_step)]
    corner_info.append(draw.Text(text_lines, 0.05, 0, time_hight/2, fill=colors.dark_beige, text_anchor='start'))

    d = draw.Group(id='map', origin=(0,0), x=0, y=0, transform='translate(1, -1)')
    d.append(draw.Rectangle(-1, -1, 2, 2, fill=colors.beige))
    # Map box
    d.append(draw.Lines(1, 1,
                        1, -1,
                        stroke_width=0.02,
                        close=False,
                        fille=colors.dark_beige,
                        stroke=colors.dark_beige))
    d.append(draw.Lines(-1, 1,
                        1, 1,
                        stroke_width=0.02,
                        close=False,
                        fille=colors.dark_beige,
                        stroke=colors.dark_beige))
    # Center cross
    d.append(draw.Lines(0.04,0,
                        -0.04, 0,
                        stroke_width=0.01,
                        close=False,
                        fille=colors.red,
                        stroke=colors.red))
    d.append(draw.Lines(0,0.04,
                        0,-0.04,
                        stroke_width=0.01,
                        close=False,
                        fille=colors.red,
                        stroke=colors.red))

    # Depot
    dep_position = drivers[0].history_move[0]
    d.append(draw.Rectangle(dep_position[0]/size - 0.025,
                            dep_position[1]/size - 0.025,
                            0.05,
                            0.05,
                            fill=colors.orange,
                            stroke_width=0.01,
                            stroke=colors.black))


    for target in targets:
        # draw two nodes for pickup and delivery + An arrow connecting them
        if target.state == -2 :
            pick_fill_col = colors.red
            pick_strok_col = colors.black
        elif target.state == -1 :
            pick_fill_col = colors.red
            pick_strok_col = colors.grey
        else :
            pick_fill_col = colors.dark_red
            pick_strok_col = colors.grey

        d.append(draw.Circle(target.pickup[0]/size,target.pickup[1]/size, 0.03,
                fill=pick_fill_col, stroke_width=0.008, stroke=pick_strok_col))
        d.append(draw.Text(str(target.identity), 0.05, target.pickup[0]/size,target.pickup[1]/size, fill=colors.dark_beige, text_anchor='end', valign='middle'))

        if target.state < 1 :
            drop_fill_col = colors.blue
            drop_strok_col = colors.black
            line_col = colors.green
        elif target.state == 1:
            drop_fill_col = colors.blue
            drop_strok_col = colors.grey
            line_col = colors.green
        else :
            drop_fill_col = colors.dark_blue
            drop_strok_col = colors.grey
            line_col = colors.dark_green

        d.append(draw.Circle(target.dropoff[0]/size,target.dropoff[1]/size, 0.03,
            fill=drop_fill_col, stroke_width=0.008, stroke=drop_strok_col))
        d.append(draw.Text(str(target.identity), 0.05, target.dropoff[0]/size,target.dropoff[1]/size, fill=colors.dark_beige, text_anchor='end', valign='middle'))
        d.append(draw.Line(target.pickup[0]/size,target.pickup[1]/size,
                           target.dropoff[0]/size,target.dropoff[1]/size,
                           stroke=line_col, stroke_width=0.008, fill='none'))

        # Target time schedule
        time_table.append(draw.Rectangle(time_margin + 1.9 * target.start_fork[0]/time_end,
                                         (target.identity-1) * time_hight / len(targets) + (time_hight / len(targets)) / 2,
                                         1.9 * (target.start_fork[1] - target.start_fork[0])/time_end,
                                         time_hight / (2*len(targets)),
                                        fill=pick_fill_col))

        time_table.append(draw.Rectangle(time_margin + 1.9 * target.end_fork[0]/time_end,
                                         (target.identity-1) * time_hight / len(targets),
                                         1.9 * (target.end_fork[1] - target.end_fork[0])/time_end,
                                         time_hight / (2*len(targets)),
                                        fill=drop_fill_col))

        target_text = 't'+str(target.identity)
        target_text_color = colors.dark_red
        if target.state == 2:
            target_text_color = colors.green
        time_table.append(draw.Text(target_text, 0.05, 1.90 + time_margin, (target.identity-1) * time_hight / len(targets), fill=target_text_color, text_anchor='start'))

    base_hight = 1.90
    driver_space = base_hight / len(drivers)
    text_lines = ['time step:' + str(time_step)]
    for driver in drivers :
        if driver.target is None :
            data_panel.append(draw.Text(str(driver.identity) + ': -| (Zzz)', 0.05, 0.15, driver_space*driver.identity, fill=colors.dark_beige, text_anchor='start'))
        else :
            if driver.order == 'picking' :
                data_panel.append(draw.Text(str(driver.identity) + ': -< (' + str(driver.target.identity) + ')', 0.05, 0.15, driver_space*driver.identity, fill=colors.dark_beige, text_anchor='start'))
            elif driver.order == 'dropping' :
                data_panel.append(draw.Text(str(driver.identity) + ': -> (' + str(driver.target.identity) + ')', 0.05, 0.15, driver_space*driver.identity, fill=colors.dark_beige, text_anchor='start'))
            elif driver.order == 'service' :
                data_panel.append(draw.Text(str(driver.identity) + ': -/\ (' + str(driver.target.identity) + ')', 0.05, 0.15, driver_space*driver.identity, fill=colors.dark_beige, text_anchor='start'))


        for i_pos in range(1, len(driver.history_move)) :
            pos1 = driver.history_move[i_pos - 1]
            pos2 = driver.history_move[i_pos]

            # Draw the historical moves
            d.append(draw.Line(pos1[0]/size, pos1[1]/size,
                               pos2[0]/size, pos2[1]/size,
                               stroke=drivers_colors_list[driver.identity - 1], stroke_width=0.005, fill='none'))

        mini_car(d, driver.position, size, driver.identity)
        mini_car(data_panel, [0.02, driver_space*driver.identity], 1, driver.identity)

        loaded = ['{']
        loaded = loaded + list(map(lambda t: str(t.identity), driver.loaded))
        loaded = '- '.join(loaded) + '}'

        data_panel.append(draw.Text(loaded, 0.05, 0, driver_space*driver.identity - 0.1, fill=colors.dark_beige, text_anchor='start'))

    # Curent time line
    time_table.append(draw.Line(time_margin + 1.9 * time_step / time_end, time_hight,
                                time_margin + 1.9 * time_step / time_end, 0,
                                stroke=colors.red, stroke_width=0.01, fill=colors.red))
    time_table.append(draw.Line(0, time_hight,
                                0, 0, stroke=colors.orange, stroke_width=0.01, fill=colors.orange))

    #d.setPixelScale(2)  # Set number of pixels per geometry unit
    # data_panel.setRenderSize(default_size/4, default_size)
    # corner_info.setRenderSize(default_size/4,(default_size/2)*time_hight)
    # time_table.setRenderSize(default_size, (default_size/2)*time_hight)
    # g = draw.Drawing(2.5, 2.5, origin='center')
    # g.setOrigin((0,-2))
    # g.append(time_table)
    # g.setOrigin((-2,0))
    # g.append(data_panel)
    #
    # # g.origin = (0,2)
    # g.setRenderSize(default_size, default_size)
    # g.append(d)
    # g.append(data_panel)
    # g.append(time_table)
    # g.append(corner_info)
    final.append(d)
    final.append(data_panel)
    final.append(corner_info)
    final.append(time_table)
    final.setRenderSize(default_size*(2.5), default_size*(2+time_hight))


    if out=='array':
        fo = tempfile.NamedTemporaryFile()
        data_panel.savePng(fo.name)
        array_data_panel = np.array(plt.imread(fo.name))
        d.savePng(fo.name)
        array_map = np.array(plt.imread(fo.name))
        time_table.savePng(fo.name)
        array_time_table = np.array(plt.imread(fo.name))
        corner_info.savePng(fo.name)
        array_corner_info = np.array(plt.imread(fo.name))
        fo.close()

        top = np.hstack([array_time_table, array_corner_info])
        botom = np.hstack([array_map, array_data_panel])
        array_image = np.vstack([top, botom])

        return array_image

    elif out=='svg':
        return final
