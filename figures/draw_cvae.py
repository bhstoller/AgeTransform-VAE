"""
Using the DrawNeuralNet library to generate a diagram of the Conditional VAE architecture.

Iqbal, Haris. PlotNeuralNet: Latex code for making neural network visualization. 2018. GitHub, https://github.com/HarisIqbal88/PlotNeuralNet. Accessed March 10.
"""

import sys

sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks import *

arch = [
    to_head('..'),
    to_cor(),
    to_begin(),

    # Input Image
    to_input('../../AgeTransform-VAE/Dataset/Team pics/Batu.jpeg', width=10, height=10, name="input"),

    # Encoder
    to_ConvConvRelu(name='enc1', s_filer=128, n_filer=(16, 16), offset="(0,0,0)", to="(0,0,0)", width=(2, 2), height=40,
                    depth=40, caption="Conv1 16x"),
    to_Pool(name="pool1", offset="(0,0,0)", to="(enc1-east)", width=1, height=32, depth=32, opacity=0.5),

    to_ConvConvRelu(name='enc2', s_filer=64, n_filer=(32, 32), offset="(1,0,0)", to="(pool1-east)", width=(3, 3),
                    height=32, depth=32, caption="Conv2 32x"),
    to_Pool(name="pool2", offset="(0,0,0)", to="(enc2-east)", width=1, height=25, depth=25, opacity=0.5),

    to_ConvConvRelu(name='enc3', s_filer=32, n_filer=(64, 64), offset="(1,0,0)", to="(pool2-east)", width=(4, 4),
                    height=25, depth=25, caption="Conv3 64x"),
    to_Pool(name="pool3", offset="(0,0,0)", to="(enc3-east)", width=1, height=16, depth=16, opacity=0.5),

    to_ConvConvRelu(name='enc4', s_filer=16, n_filer=(128, 128), offset="(1,0,0)", to="(pool3-east)", width=(5, 5),
                    height=16, depth=16, caption="Conv4 128x"),
    to_Pool(name="pool4", offset="(0,0,0)", to="(enc4-east)", width=1, height=8, depth=8, opacity=0.5),

    # Latent Space (Mu and LogVar)
    to_ConvConvRelu(name='mu', s_filer=8, n_filer=(256, 256), offset="(2,1,0)", to="(pool4-east)", width=(8, 8),
                    height=8, depth=8, caption="Mu"),
    to_ConvConvRelu(name='logvar', s_filer=8, n_filer=(256, 256), offset="(2,-1,0)", to="(pool4-east)", width=(8, 8),
                    height=8, depth=8, caption="LogVar"),
    to_skip(of="pool4", to="mu", pos=1.25),
    to_skip(of="pool4", to="logvar", pos=1.25),

    to_SoftMax(name='condition', s_filer=1, offset="(16,4,0)", to="(input)", width=0.5, height=2, depth=2,
                   caption="Condition"),

    # Decoder

    *block_Unconv(name="dec1", botton="mu", top="up1", s_filer=16, n_filer=128, offset="(2.1,0,0)", size=(16, 16, 5.5),
                  opacity=0.5),

    to_skip(of="condition", to="unpool_dec1", pos=1.25),

    *block_Unconv(name="dec2", botton="up1", top="up2", s_filer=32, n_filer=64, offset="(2.1,0,0)", size=(25, 25, 4.5),
                  opacity=0.5),
    *block_Unconv(name="dec3", botton="up2", top="up3", s_filer=64, n_filer=32, offset="(2.1,0,0)", size=(32, 32, 3.5),
                  opacity=0.5),
    *block_Unconv(name="dec4", botton="up3", top="up4", s_filer=128, n_filer=16, offset="(2.1,0,0)", size=(40, 40, 2.5),
                  opacity=0.5),

    # Output
    to_ConvSoftMax(name="output", s_filer=128, offset="(0.75,0,0)", to="(up4-east)", width=1, height=40, depth=40,
                   caption="Reconstructed Image"),
    to_connection("up4", "output"),

    to_end()
]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')


if __name__ == '__main__':
    main()