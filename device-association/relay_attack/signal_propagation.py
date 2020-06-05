from __future__ import division
import os
import numpy
import subprocess
import scipy.interpolate
import matplotlib.pyplot as plt
import utils.statistics as statistics
import coupling.relay_attack.outlier as outlier
import coupling.relay_attack.analysis as analysis

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

def get_measurement_coordinates(
        x_size=25, y_size=5, start_pos=1, end_pos=119, breakpoints=[3, 116], scale=1):
    
    end = False
    position = start_pos
    measurement_coordinates = dict()
    for x in range(0, x_size * scale, scale):
        for y in range(0, y_size * scale, scale):
            measurement_coordinates[position] = (x, y)
            is_breakpoint = False
            if position in breakpoints:
                is_breakpoint = True
            elif position == end_pos:
                end = True
            position += 1
            if end or is_breakpoint:
                break
        if end:
            break
    return measurement_coordinates

def convert_to_3d(measurements_map, fill_value):
    measurement_coordinates = get_measurement_coordinates()
    x = numpy.arange(max([x for x, _ in measurement_coordinates.values()]) + 1)
    y = numpy.arange(max([y for _, y in measurement_coordinates.values()]) + 1)
    z = numpy.full((x.size, y.size), fill_value)
    for pos in measurements_map:
        z[measurement_coordinates[pos]] = measurements_map[pos]
    return x, y, z

def convert_from_svg_to_emf(
        path_svg_file, delete_svg_file=False, inkscape_path="C://Program Files//Inkscape//inkscape.exe"):
    
    path_emf_file = path_svg_file[:path_svg_file.rfind(".") + 1] + "emf"
    subprocess.call([inkscape_path, path_svg_file, "--export-emf", path_emf_file])
    if delete_svg_file:
        os.remove(path_svg_file)
    
def interpolate(x, y, z, delta=0.01):
    xi = numpy.arange(x[0], x[-1] + delta * 0.1, delta)
    yi = numpy.arange(y[0], y[-1] + delta * 0.1, delta)
    xi, yi = numpy.meshgrid(xi, yi)
    return scipy.interpolate.interpn((x, y), z, (xi, yi), method="splinef2d")

def plot_heatmap(
        interpolated_z, z_bottom_limit, z_top_limit, ylabel, color, filename, result_directory, plot_format="svg"):
    
    interpolated_z[interpolated_z > z_top_limit] = z_top_limit
    interpolated_z[interpolated_z < z_bottom_limit] = z_bottom_limit
    fig, ax = plt.subplots()
    ax.set_axis_off()
    im = ax.imshow(interpolated_z, cmap=color)
    bar = fig.colorbar(im, ticks=numpy.linspace(z_bottom_limit, z_top_limit, 6))
    bar.set_label(ylabel)
    filepath = os.path.join(result_directory, filename + "." + plot_format)
    fig.savefig(filepath, format=plot_format, bbox_inches="tight")
    #plt.show()
    plt.close(fig)
    convert_from_svg_to_emf(filepath)
    
def plot_measurement_statistics(
        testbed, measurement_positions, median, mean, std, entropy, ylabel, filename, result_directory, plot_format="pdf"):
    
    xlabel = { "line-of-sight": "Distance (m)", "non-line-of-sight": "Room" }
    xlabel = xlabel[testbed]
    x = measurement_positions
    if "distance" in xlabel.lower():  # convert position to distance
        distance_between_position = 2
        x = range(distance_between_position,
                  (len(measurement_positions) * distance_between_position) + 1,
                  distance_between_position)
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    ax1.errorbar(x, mean, numpy.array([[0] * len(std), std]), capsize=5, capthick=3, lw=3, label="Mean +/- Std.", color="blue")
    ax2.plot(x, median, label="Median", color="green")
    ax3.plot(x, entropy, label="Entropy", color="red")
    ax3.set_xticks(x[::3])
    ax3.set_xlabel(xlabel)
    fig.text(-0.02, 0.5, ylabel, va="center", rotation="vertical")
    fig.legend()
    fig.set_figwidth(fig.get_figwidth() * 1.1)
    filepath = os.path.join(result_directory, filename + "." + plot_format)
    fig.savefig(filepath, format=plot_format, bbox_inches="tight")
    #plt.show()
    plt.close(fig)
    
def summarize_per_measurement_point(measurements, conversion_factor=1):
    measurements_position = numpy.array(sorted(measurements.keys()))
    num_measurements = len(measurements_position)
    median = numpy.empty(num_measurements)
    mean = numpy.empty(num_measurements)
    std = numpy.empty(num_measurements)
    entropy = numpy.empty(num_measurements)
    rawdatalen = numpy.empty(num_measurements)
    nonzerolen = numpy.empty(num_measurements)
    outlierlen = numpy.empty(num_measurements)
    for i, pos in enumerate(measurements_position):
        data = numpy.array(measurements[pos])
        rawdatalen[i] = len(data)
        data = data[numpy.nonzero(data)]
        nonzerolen[i] = len(data)
        entropy[i] = statistics.entropy(data)
        outliers = outlier.detect_by_z_score_scipy(data, 2)
        outlierlen[i] = len(outliers[0])
        median[i] = numpy.median(data)
        mean[i] = numpy.mean(data)
        std[i] = numpy.std(data)
    median *= conversion_factor
    mean *= conversion_factor
    std *= conversion_factor
    return measurements_position, median, mean, std, entropy, rawdatalen, nonzerolen, outlierlen

def process_signal_propagation(
        testbed, measurements, conversion_factor, fill_value, ylabel, color, filename, result_directory):
    
    measurement_positions, median, mean, std, entropy, rawdatalen, nonzerolen, outlierlen = summarize_per_measurement_point(
        measurements, conversion_factor)
    
    fill_value = fill_value if fill_value != None else numpy.max(median)
    median_map = dict(zip(measurement_positions, median))
    x, y, z = convert_to_3d(median_map, fill_value)
    z_interpolated = interpolate(x, y, z)
    plot_heatmap(
        z_interpolated, numpy.min(z), numpy.max(z), ylabel, color, filename + "-signal-propagation", result_directory)
    
    if "non-line-of-sight" in testbed:
        measurements_per_room = analysis.join_measurements_per_room(measurements)
        measurement_positions, median, mean, std, entropy, rawdatalen, nonzerolen, outlierlen = summarize_per_measurement_point(
            measurements_per_room, conversion_factor)
    
    print(testbed)
    print(ylabel)
    print("entropy: {0:.2f}+/-{1:.2f}".format(numpy.mean(entropy), numpy.std(entropy)))
    print("ratio non-zero/raw data: {0:.2f}".format(sum(nonzerolen) / sum(rawdatalen)))
    print("ratio outlier/non-zero data: {0:.2f}".format(sum(outlierlen) / sum(nonzerolen)))
    
    plot_measurement_statistics(
        testbed, measurement_positions, median, mean, std, entropy, ylabel, filename + "-measurement-statistics", result_directory)
    
def main():
    for testbed in analysis.testbeds:
        result_directory = os.path.join(analysis.result_base_path, testbed)
        if not os.path.exists(result_directory):
            os.makedirs(result_directory)
        
        latency_measurements = analysis.load_latency_data(analysis.measurement_directory, testbed)
        process_signal_propagation(
            testbed, latency_measurements, 1000, None, "Latency (ms)", "Spectral_r", "latency", result_directory)
        
        signal_strength_measurements = analysis.load_rssi_data(analysis.measurement_directory, testbed)
        process_signal_propagation(
            testbed, signal_strength_measurements, 1, -100, "RSSI", "Spectral", "rssi", result_directory)
        
if __name__ == "__main__":
    main()
    