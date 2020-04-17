import torch
import copy


def conv_mask(conv, in_mask, out_mask, device='cuda'):
    """
    :param conv: original conv layer
    :param in_mask: boolean mask
    :param out_mask: boolean mask
    :param device: cuda / cpu
    :return:
    """
    in_channels = sum(in_mask)
    out_channels = sum(out_mask)

    print(f"conv2d in channels : {conv.in_channels} -> {in_channels}")
    print(f"conv2d out channels : {conv.out_channels} -> {out_channels}")

    new_conv = torch.nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=conv.kernel_size,
                               stride=conv.stride,
                               padding=conv.padding,
                               dilation=conv.dilation,
                               groups=conv.groups,
                               bias=(conv.bias is not None)).to(device)

    new_conv.weight.data = conv.weight.data[in_mask, :]
    new_conv.weight.data = new_conv.weight.data[:, out_mask]

    return new_conv


def conv_pre_mask(conv, mask, device='cuda'):
    """
    :param conv: original conv layer
    :param mask: boolean mask
    :param device: cuda / cpu
    :return:
    """

    in_channels = sum(mask)
    print(f"conv2d in channels : {conv.in_channels} -> {in_channels}")

    new_conv = torch.nn.Conv2d(in_channels=in_channels,
                               out_channels=conv.out_channels,
                               kernel_size=conv.kernel_size,
                               stride=conv.stride,
                               padding=conv.padding,
                               dilation=conv.dilation,
                               groups=conv.groups,
                               bias=(conv.bias is not None)).to(device)

    new_conv.weight.data = conv.weight.data[:, mask]

    return new_conv


def conv_post_mask(conv, mask, device='cuda'):
    """
    :param conv: original conv layer
    :param mask: boolean mask
    :param device: cuda / cpu
    :return:
    """

    out_channels = sum(mask)
    print(f"conv2d out channels : {conv.out_channels} -> {out_channels}")

    new_conv = torch.nn.Conv2d(in_channels=conv.in_channels,
                               out_channels=out_channels,
                               kernel_size=conv.kernel_size,
                               stride=conv.stride,
                               padding=conv.padding,
                               dilation=conv.dilation,
                               groups=conv.groups,
                               bias=(conv.bias is not None)).to(device)

    new_conv.weight.data = conv.weight.data[mask, :]

    return new_conv


def bn2d_mask(bn, mask):
    """
    :param bn: original bn layer
    :param mask: boolean mask
    :return:
    """

    new_bn = copy.deepcopy(bn)
    new_bn.weight.data = new_bn.weight.data[mask]
    new_bn.bias.data = new_bn.bias.data[mask]
    new_bn.running_mean.data = new_bn.running_mean.data[mask]
    new_bn.running_var.data = new_bn.running_var.data[mask]

    return new_bn


def linear_mask(linear, in_mask, out_mask, last_feature=(7, 7), device='cuda'):
    """
    :param linear: original linear layer
    :param in_mask: boolean mask (conv)
    :param out_mask: boolean mask (linear)
    :param device: cuda / cpu
    :return:
    """
    in_features = sum(in_mask) * last_feature[0] * last_feature[1]
    out_features = sum(out_mask)

    print(f"linear in features : {linear.in_features} -> {in_features}")
    print(f"linear out features : {linear.out_features} -> {out_features}")

    new_linear = torch.nn.Linear(in_features=in_features,
                                 out_features=out_features,
                                 bias=(linear.bias is not None)).to(device)

    linear_in_mask = []

    for i in in_mask:
        linear_in_mask += [i] * last_feature[0] * last_feature[1]

    new_linear.weight.data = linear.weight.data[out_mask]
    new_linear.weight.data = new_linear.weight.data[:, linear_in_mask]
    new_linear.bias.data = linear.bias.data[out_mask]

    return new_linear


def linear_pre_mask(linear, mask, device='cuda'):
    """
        :param linear: original linear layer
        :param mask: boolean mask (linear)
        :param device: cuda / cpu
        :return:
        """
    in_features = sum(mask)

    print(f"linear in features : {linear.in_features} -> {in_features}")

    new_linear = torch.nn.Linear(in_features=in_features,
                                 out_features=linear.out_features,
                                 bias=(linear.bias is not None)).to(device)

    new_linear.weight.data = linear.weight.data[:, mask]
    new_linear.bias.data = linear.bias.data

    return new_linear


def linear_post_mask(linear, mask, device='cuda'):
    """
    :param linear: original linear layer
    :param mask: boolean mask (linear)
    :param device: cuda / cpu
    :return:
    """
    out_features = sum(mask)

    print(f"linear out features : {linear.out_features} -> {out_features}")

    new_linear = torch.nn.Linear(in_features=linear.in_features,
                                 out_features=out_features,
                                 bias=(linear.bias is not None)).to(device)

    new_linear.weight.data = linear.weight.data[mask]
    new_linear.bias.data = linear.bias.data[mask]

    return new_linear
