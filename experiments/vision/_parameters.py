def parameters(channels, width_expansions):
    data_channels = 3
    kernel_size = 9
    encoder_conv = kernel_size * data_channels * channels[0]
    res = 0
    for i in range(len(channels)):
        res += kernel_size * channels[i] * channels[i] * width_expansions[i] * 2

    down = 0
    for i in range(len(channels) - 1):
        down += kernel_size * channels[i] * channels[i + 1]
        down += kernel_size * channels[i + 1] * channels[i + 1]
        down += channels[i] * channels[i + 1]

    final_linear = channels[-1] * 10 + 10
    count = encoder_conv + res + down + final_linear
    return count


if __name__ == "__main__":
    channels = [128, 256, 256, 512]
    widths = [1, 3, 3, 1]
    count = parameters(channels, widths)
    print(f"Parameters: {count:.3e}")
