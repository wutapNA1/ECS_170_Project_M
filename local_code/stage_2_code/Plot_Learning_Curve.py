'''
Utility module that draws the learning convergence curves of a Stage 2 run.

The per-epoch numbers recorded in Method_MLP.history are always written to a CSV file,
and are drawn into a PNG when a drawing library is available: matplotlib is preferred and
Pillow is used as a fallback, so the plots can be generated on a machine that only has one
of the two installed (or neither, in which case the CSV is still produced).
'''

import csv
import importlib.util
import math
import os

# matplotlib colors: blue for the training curve, red for the testing curve
_TRAIN_COLOR = (31, 119, 180)
_TEST_COLOR = (214, 39, 40)

# the two panels of the figure: (panel title, key of the training series, key of the testing series)
_PANELS = [('loss', 'train_loss', 'test_loss'), ('accuracy', 'train_accuracy', 'test_accuracy')]


def _has_module(name):
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def save_history_csv(history, csv_path):
    '''Write the per epoch training / testing numbers to a CSV file'''
    folder = os.path.dirname(csv_path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
    keys = [key for key in ['epoch', 'train_loss', 'train_accuracy', 'test_loss', 'test_accuracy']
            if key in history]
    length = max([len(history[key]) for key in keys] + [0])
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for index in range(length):
            row = []
            for key in keys:
                values = history[key]
                row.append(values[index] if index < len(values) else '')
            writer.writerow(row)
    return csv_path


def save_learning_curve(history, output_folder, file_stem='learning_curve', title=None):
    '''
    Draw the learning convergence curves (loss and accuracy, training and testing set).

    Returns a dictionary with the paths of the generated files, the 'png' entry is None
    when no drawing library is installed.
    '''
    if not output_folder:
        output_folder = '.'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    csv_path = save_history_csv(history, os.path.join(output_folder, file_stem + '_history.csv'))
    png_path = os.path.join(output_folder, file_stem + '.png')

    if _has_module('matplotlib'):
        try:
            _draw_with_matplotlib(history, png_path, title)
            print('  learning convergence curve saved to ' + png_path)
            return {'png': png_path, 'csv': csv_path}
        except Exception as error:
            print('  matplotlib plotting failed (' + str(error) + '), trying Pillow')

    if _has_module('PIL'):
        try:
            _draw_with_pillow(history, png_path, title)
            print('  learning convergence curve saved to ' + png_path)
            return {'png': png_path, 'csv': csv_path}
        except Exception as error:
            print('  Pillow plotting failed (' + str(error) + ')')

    print('  neither matplotlib nor Pillow is installed, the per epoch history was saved to ' + csv_path)
    return {'png': None, 'csv': csv_path}


# --------------------------------------------------------------------------------------
# matplotlib version
# --------------------------------------------------------------------------------------
def _draw_with_matplotlib(history, png_path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    epochs = list(history['epoch'])
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, (panel_title, train_key, test_key) in zip(axes, _PANELS):
        train_values = [float(value) for value in history[train_key]]
        axis.plot(epochs, train_values, marker='o', color='#1f77b4', label='training set')
        test_values = [float(value) for value in history.get(test_key, [])]
        if test_values:
            axis.plot(epochs[:len(test_values)], test_values, marker='s', color='#d62728',
                      label='testing set')
        axis.set_title('model ' + panel_title)
        axis.set_xlabel('epoch')
        axis.set_ylabel(panel_title)
        axis.grid(True, alpha=0.3)
        axis.legend()
    if title:
        figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(png_path, dpi=150)
    plt.close(figure)


# --------------------------------------------------------------------------------------
# Pillow version (used when matplotlib is not installed)
# --------------------------------------------------------------------------------------
def _pillow_font(size):
    from PIL import ImageFont
    for name in ['arial.ttf', 'DejaVuSans.ttf', 'segoeui.ttf']:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def _pillow_text(draw, x, y, text, font, fill=(0, 0, 0), center=False, right=False):
    try:
        left, top, right_edge, bottom = draw.textbbox((0, 0), text, font=font)
        text_width = right_edge - left
    except Exception:
        text_width = draw.textsize(text, font=font)[0]
        left, top = 0, 0
    if center:
        x = x - text_width / 2.0
    if right:
        x = x - text_width
    draw.text((x - left, y - top), text, font=font, fill=fill)


def _nice_step(span, target=6):
    '''Round the distance between two ticks up to 1, 2 or 5 times a power of ten'''
    if span <= 0:
        return 1.0
    raw = span / float(target)
    exponent = int(math.floor(math.log10(raw)))
    fraction = raw / (10.0 ** exponent)
    for candidate in (1.0, 2.0, 5.0, 10.0):
        if fraction <= candidate + 1e-9:
            return candidate * (10.0 ** exponent)
    return 10.0 * (10.0 ** exponent)


def _draw_with_pillow(history, png_path, title):
    from PIL import Image, ImageDraw

    width, height = 1280, 540
    image = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    font_title = _pillow_font(20)
    font_label = _pillow_font(15)
    font_tick = _pillow_font(13)

    if title:
        _pillow_text(draw, width / 2.0, 16, str(title), font_title, center=True)

    boxes = [(150, 70, 640, 460), (790, 70, 1270, 460)]
    epochs = [float(value) for value in history['epoch']]

    for (panel_title, train_key, test_key), box in zip(_PANELS, boxes):
        left, top, right, bottom = box
        train_values = [float(value) for value in history.get(train_key, [])]
        test_values = [float(value) for value in history.get(test_key, [])]
        all_values = [value for value in train_values + test_values if value == value]
        if not all_values:
            continue

        low, high = min(all_values), max(all_values)
        if high - low < 1e-9:
            low, high = low - 0.5, high + 0.5
        padding = (high - low) * 0.08
        low -= padding
        high += padding
        if panel_title == 'accuracy':
            # an accuracy can never be above 1, so there is no point in drawing above it
            high = min(high, 1.0)
        else:
            # a loss can never be below 0, so there is no point in drawing below it
            low = max(low, 0.0)
        if high - low < 1e-9:
            high = low + 1.0

        # round the axis limits and the distance between two ticks to nice values
        tick = _nice_step(high - low)
        low = math.floor(round(low / tick, 9)) * tick
        high = math.ceil(round(high / tick, 9)) * tick
        n_tick = max(1, int(round((high - low) / tick)))
        decimals = max(0, -int(math.floor(math.log10(tick) + 1e-9)))

        # light grid: one line per tick on the y axis, 5 reference lines on the x axis
        for index in range(n_tick + 1):
            y = top + (bottom - top) * index / float(n_tick)
            draw.line([(left, y), (right, y)], fill=(225, 225, 225), width=1)
        for index in range(5):
            x = left + (right - left) * index / 4.0
            draw.line([(x, top), (x, bottom)], fill=(225, 225, 225), width=1)

        # y axis tick labels
        for index in range(n_tick + 1):
            value = high - tick * index
            if abs(value) < tick * 1e-6:
                value = 0.0
            text = '%.*f' % (decimals, value)
            y = top + (bottom - top) * index / float(n_tick)
            _pillow_text(draw, left - 10, y - 7, text, font_tick, fill=(70, 70, 70), right=True)

        # x axis tick labels
        n_epoch = max(len(epochs) - 1, 1)
        tick_step = max(1, int(len(epochs) / 8.0)) if len(epochs) > 8 else 1
        for index in range(0, len(epochs), tick_step):
            x = left + (right - left) * index / n_epoch
            _pillow_text(draw, x, bottom + 8, str(int(epochs[index])), font_tick,
                         fill=(70, 70, 70), center=True)

        # axes
        draw.rectangle([left, top, right, bottom], outline=(90, 90, 90), width=2)

        # curve of one series
        def draw_series(values, color):
            points = []
            for index, value in enumerate(values):
                if value != value:      # skip NaN
                    continue
                x = left + (right - left) * index / n_epoch
                y = bottom - (bottom - top) * (value - low) / (high - low)
                points.append((x, y))
            if len(points) > 1:
                draw.line(points, fill=color, width=3, joint='curve')
            for x, y in points:
                draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=color)

        draw_series(train_values, _TRAIN_COLOR)
        draw_series(test_values, _TEST_COLOR)

        # legend: the loss curves fall towards the right and the accuracy curves rise, so
        # the legend is put in the corner that stays empty, on a white background
        legend_x = right - 140
        legend_y = top + 12 if panel_title == 'loss' else bottom - 58
        draw.rectangle([legend_x - 8, legend_y - 8, right - 4, legend_y + 46], fill=(255, 255, 255))
        for label, color in [('training set', _TRAIN_COLOR), ('testing set', _TEST_COLOR)]:
            draw.line([(legend_x, legend_y + 7), (legend_x + 26, legend_y + 7)], fill=color, width=3)
            _pillow_text(draw, legend_x + 34, legend_y, label, font_label)
            legend_y += 22

        # panel title and x axis label
        _pillow_text(draw, (left + right) / 2.0, top - 32, 'model ' + panel_title,
                     font_title, center=True)
        _pillow_text(draw, (left + right) / 2.0, bottom + 34, 'epoch', font_label, center=True)

    image.save(png_path)
