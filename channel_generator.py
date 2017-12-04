#!/usr/bin/env python2
# -*- coding: utf-8 -*-
##################################################
# GNU Radio Python Flow Graph
# Title: Channel Generator
# Generated: Fri Dec  1 21:02:30 2017
##################################################

from gnuradio import analog
from gnuradio import blocks
from gnuradio import channels
from gnuradio import digital
from gnuradio import eng_notation
from gnuradio import fft
from gnuradio import gr
from gnuradio.eng_option import eng_option
from gnuradio.fft import window
from gnuradio.filter import firdes
from optparse import OptionParser
import numpy


class channel_generator(gr.top_block):

    def __init__(self, PSKMod1=2, PSKMod2=2, QAMMod3=64, QAMMod4=64, amplitude1=20, amplitude2=20, amplitude3=20, amplitude4=20, bw1=2, bw2=2, bw3=10, bw4=10, cf1=920, cf2=920, cf3=920, cf4=920, file_name='default_file.dat', noiselevel=.01, on_off1=0, on_off2=0, on_off3=0, on_off4=0):
        gr.top_block.__init__(self, "Channel Generator")

        ##################################################
        # Parameters
        ##################################################
        self.PSKMod1 = PSKMod1
        self.PSKMod2 = PSKMod2
        self.QAMMod3 = QAMMod3
        self.QAMMod4 = QAMMod4
        self.amplitude1 = amplitude1
        self.amplitude2 = amplitude2
        self.amplitude3 = amplitude3
        self.amplitude4 = amplitude4
        self.bw1 = bw1
        self.bw2 = bw2
        self.bw3 = bw3
        self.bw4 = bw4
        self.cf1 = cf1
        self.cf2 = cf2
        self.cf3 = cf3
        self.cf4 = cf4
        self.file_name = file_name
        self.noiselevel = noiselevel
        self.on_off1 = on_off1
        self.on_off2 = on_off2
        self.on_off3 = on_off3
        self.on_off4 = on_off4

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 30720
        self.bandwidth = bandwidth = 30e6
        self.spreading_ratio = spreading_ratio = samp_rate/(bandwidth/1000000)
        self.offset = offset = 915
        self.center_freq = center_freq = 915000000

        ##################################################
        # Blocks
        ##################################################
        self.fft_vxx_0 = fft.fft_vcc(1024, True, (window.blackmanharris(1024)), False, 1)
        self.digital_qam_mod_0_0 = digital.qam.qam_mod(
          constellation_points=QAMMod4,
          mod_code="gray",
          differential=True,
          samples_per_symbol=40/bw4,
          excess_bw=0.35,
          verbose=False,
          log=False,
          )
        self.digital_qam_mod_0 = digital.qam.qam_mod(
          constellation_points=QAMMod3,
          mod_code="gray",
          differential=True,
          samples_per_symbol=40/bw3,
          excess_bw=0.35,
          verbose=False,
          log=False,
          )
        self.digital_psk_mod_0_0 = digital.psk.psk_mod(
          constellation_points=PSKMod2,
          mod_code="gray",
          differential=True,
          samples_per_symbol=40/bw2,
          excess_bw=0.35,
          verbose=False,
          log=False,
          )
        self.digital_psk_mod_0 = digital.psk.psk_mod(
          constellation_points=PSKMod1,
          mod_code="gray",
          differential=True,
          samples_per_symbol=40/bw1,
          excess_bw=0.35,
          verbose=False,
          log=False,
          )
        self.channels_channel_model_0 = channels.channel_model(
        	noise_voltage=noiselevel,
        	frequency_offset=0,
        	epsilon=1.0,
        	taps=(1.0 + 1.0j, ),
        	noise_seed=0,
        	block_tags=False
        )
        self.blocks_throttle_1_2 = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate,True)
        self.blocks_throttle_1_1 = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate,True)
        self.blocks_throttle_1_0 = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate,True)
        self.blocks_throttle_1 = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate,True)
        self.blocks_throttle_0_2 = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate,True)
        self.blocks_throttle_0_1 = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate,True)
        self.blocks_throttle_0_0 = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate,True)
        self.blocks_throttle_0 = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate,True)
        self.blocks_stream_to_vector_0 = blocks.stream_to_vector(gr.sizeof_gr_complex*1, 1024)
        self.blocks_multiply_xx_0_2 = blocks.multiply_vcc(1)
        self.blocks_multiply_xx_0_1 = blocks.multiply_vcc(1)
        self.blocks_multiply_xx_0_0 = blocks.multiply_vcc(1)
        self.blocks_multiply_xx_0 = blocks.multiply_vcc(1)
        self.blocks_file_sink_0_0 = blocks.file_sink(gr.sizeof_gr_complex*1024, file_name, False)
        self.blocks_file_sink_0_0.set_unbuffered(False)
        self.blocks_add_xx_0 = blocks.add_vcc(1)
        self.analog_sig_source_x_0_2 = analog.sig_source_c(samp_rate, analog.GR_COS_WAVE, (cf4-offset)*spreading_ratio, amplitude4, 0)
        self.analog_sig_source_x_0_1 = analog.sig_source_c(samp_rate, analog.GR_COS_WAVE, (cf3-offset)*spreading_ratio, amplitude3, 0)
        self.analog_sig_source_x_0_0 = analog.sig_source_c(samp_rate, analog.GR_COS_WAVE, (cf2-offset)*spreading_ratio, amplitude2, 0)
        self.analog_sig_source_x_0 = analog.sig_source_c(samp_rate, analog.GR_COS_WAVE, (cf1-offset)*spreading_ratio, amplitude1, 0)
        self.analog_random_source_x_0 = blocks.vector_source_b(map(int, numpy.random.randint(0, 2, 1000)), True)
        self.analog_const_source_x_0_2 = analog.sig_source_c(0, analog.GR_CONST_WAVE, 0, 0, on_off4)
        self.analog_const_source_x_0_1 = analog.sig_source_c(0, analog.GR_CONST_WAVE, 0, 0, on_off3)
        self.analog_const_source_x_0_0 = analog.sig_source_c(0, analog.GR_CONST_WAVE, 0, 0, on_off2)
        self.analog_const_source_x_0 = analog.sig_source_c(0, analog.GR_CONST_WAVE, 0, 0, on_off1)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_const_source_x_0, 0), (self.blocks_multiply_xx_0, 2))
        self.connect((self.analog_const_source_x_0_0, 0), (self.blocks_multiply_xx_0_0, 2))
        self.connect((self.analog_const_source_x_0_1, 0), (self.blocks_multiply_xx_0_1, 2))
        self.connect((self.analog_const_source_x_0_2, 0), (self.blocks_multiply_xx_0_2, 2))
        self.connect((self.analog_random_source_x_0, 0), (self.digital_psk_mod_0, 0))
        self.connect((self.analog_random_source_x_0, 0), (self.digital_psk_mod_0_0, 0))
        self.connect((self.analog_random_source_x_0, 0), (self.digital_qam_mod_0, 0))
        self.connect((self.analog_random_source_x_0, 0), (self.digital_qam_mod_0_0, 0))
        self.connect((self.analog_sig_source_x_0, 0), (self.blocks_throttle_0, 0))
        self.connect((self.analog_sig_source_x_0_0, 0), (self.blocks_throttle_0_0, 0))
        self.connect((self.analog_sig_source_x_0_1, 0), (self.blocks_throttle_0_1, 0))
        self.connect((self.analog_sig_source_x_0_2, 0), (self.blocks_throttle_0_2, 0))
        self.connect((self.blocks_add_xx_0, 0), (self.channels_channel_model_0, 0))
        self.connect((self.blocks_multiply_xx_0, 0), (self.blocks_add_xx_0, 0))
        self.connect((self.blocks_multiply_xx_0_0, 0), (self.blocks_add_xx_0, 1))
        self.connect((self.blocks_multiply_xx_0_1, 0), (self.blocks_add_xx_0, 2))
        self.connect((self.blocks_multiply_xx_0_2, 0), (self.blocks_add_xx_0, 3))
        self.connect((self.blocks_stream_to_vector_0, 0), (self.fft_vxx_0, 0))
        self.connect((self.blocks_throttle_0, 0), (self.blocks_multiply_xx_0, 1))
        self.connect((self.blocks_throttle_0_0, 0), (self.blocks_multiply_xx_0_0, 1))
        self.connect((self.blocks_throttle_0_1, 0), (self.blocks_multiply_xx_0_1, 1))
        self.connect((self.blocks_throttle_0_2, 0), (self.blocks_multiply_xx_0_2, 1))
        self.connect((self.blocks_throttle_1, 0), (self.blocks_multiply_xx_0, 0))
        self.connect((self.blocks_throttle_1_0, 0), (self.blocks_multiply_xx_0_0, 0))
        self.connect((self.blocks_throttle_1_1, 0), (self.blocks_multiply_xx_0_1, 0))
        self.connect((self.blocks_throttle_1_2, 0), (self.blocks_multiply_xx_0_2, 0))
        self.connect((self.channels_channel_model_0, 0), (self.blocks_stream_to_vector_0, 0))
        self.connect((self.digital_psk_mod_0, 0), (self.blocks_throttle_1, 0))
        self.connect((self.digital_psk_mod_0_0, 0), (self.blocks_throttle_1_0, 0))
        self.connect((self.digital_qam_mod_0, 0), (self.blocks_throttle_1_1, 0))
        self.connect((self.digital_qam_mod_0_0, 0), (self.blocks_throttle_1_2, 0))
        self.connect((self.fft_vxx_0, 0), (self.blocks_file_sink_0_0, 0))

    def get_PSKMod1(self):
        return self.PSKMod1

    def set_PSKMod1(self, PSKMod1):
        self.PSKMod1 = PSKMod1

    def get_PSKMod2(self):
        return self.PSKMod2

    def set_PSKMod2(self, PSKMod2):
        self.PSKMod2 = PSKMod2

    def get_QAMMod3(self):
        return self.QAMMod3

    def set_QAMMod3(self, QAMMod3):
        self.QAMMod3 = QAMMod3

    def get_QAMMod4(self):
        return self.QAMMod4

    def set_QAMMod4(self, QAMMod4):
        self.QAMMod4 = QAMMod4

    def get_amplitude1(self):
        return self.amplitude1

    def set_amplitude1(self, amplitude1):
        self.amplitude1 = amplitude1
        self.analog_sig_source_x_0.set_amplitude(self.amplitude1)

    def get_amplitude2(self):
        return self.amplitude2

    def set_amplitude2(self, amplitude2):
        self.amplitude2 = amplitude2
        self.analog_sig_source_x_0_0.set_amplitude(self.amplitude2)

    def get_amplitude3(self):
        return self.amplitude3

    def set_amplitude3(self, amplitude3):
        self.amplitude3 = amplitude3
        self.analog_sig_source_x_0_1.set_amplitude(self.amplitude3)

    def get_amplitude4(self):
        return self.amplitude4

    def set_amplitude4(self, amplitude4):
        self.amplitude4 = amplitude4
        self.analog_sig_source_x_0_2.set_amplitude(self.amplitude4)

    def get_bw1(self):
        return self.bw1

    def set_bw1(self, bw1):
        self.bw1 = bw1

    def get_bw2(self):
        return self.bw2

    def set_bw2(self, bw2):
        self.bw2 = bw2

    def get_bw3(self):
        return self.bw3

    def set_bw3(self, bw3):
        self.bw3 = bw3

    def get_bw4(self):
        return self.bw4

    def set_bw4(self, bw4):
        self.bw4 = bw4

    def get_cf1(self):
        return self.cf1

    def set_cf1(self, cf1):
        self.cf1 = cf1
        self.analog_sig_source_x_0.set_frequency((self.cf1-self.offset)*self.spreading_ratio)

    def get_cf2(self):
        return self.cf2

    def set_cf2(self, cf2):
        self.cf2 = cf2
        self.analog_sig_source_x_0_0.set_frequency((self.cf2-self.offset)*self.spreading_ratio)

    def get_cf3(self):
        return self.cf3

    def set_cf3(self, cf3):
        self.cf3 = cf3
        self.analog_sig_source_x_0_1.set_frequency((self.cf3-self.offset)*self.spreading_ratio)

    def get_cf4(self):
        return self.cf4

    def set_cf4(self, cf4):
        self.cf4 = cf4
        self.analog_sig_source_x_0_2.set_frequency((self.cf4-self.offset)*self.spreading_ratio)

    def get_file_name(self):
        return self.file_name

    def set_file_name(self, file_name):
        self.file_name = file_name
        self.blocks_file_sink_0_0.open(self.file_name)

    def get_noiselevel(self):
        return self.noiselevel

    def set_noiselevel(self, noiselevel):
        self.noiselevel = noiselevel
        self.channels_channel_model_0.set_noise_voltage(self.noiselevel)

    def get_on_off1(self):
        return self.on_off1

    def set_on_off1(self, on_off1):
        self.on_off1 = on_off1
        self.analog_const_source_x_0.set_offset(self.on_off1)

    def get_on_off2(self):
        return self.on_off2

    def set_on_off2(self, on_off2):
        self.on_off2 = on_off2
        self.analog_const_source_x_0_0.set_offset(self.on_off2)

    def get_on_off3(self):
        return self.on_off3

    def set_on_off3(self, on_off3):
        self.on_off3 = on_off3
        self.analog_const_source_x_0_1.set_offset(self.on_off3)

    def get_on_off4(self):
        return self.on_off4

    def set_on_off4(self, on_off4):
        self.on_off4 = on_off4
        self.analog_const_source_x_0_2.set_offset(self.on_off4)

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.set_spreading_ratio(self.samp_rate/(self.bandwidth/1000000))
        self.blocks_throttle_1_2.set_sample_rate(self.samp_rate)
        self.blocks_throttle_1_1.set_sample_rate(self.samp_rate)
        self.blocks_throttle_1_0.set_sample_rate(self.samp_rate)
        self.blocks_throttle_1.set_sample_rate(self.samp_rate)
        self.blocks_throttle_0_2.set_sample_rate(self.samp_rate)
        self.blocks_throttle_0_1.set_sample_rate(self.samp_rate)
        self.blocks_throttle_0_0.set_sample_rate(self.samp_rate)
        self.blocks_throttle_0.set_sample_rate(self.samp_rate)
        self.analog_sig_source_x_0_2.set_sampling_freq(self.samp_rate)
        self.analog_sig_source_x_0_1.set_sampling_freq(self.samp_rate)
        self.analog_sig_source_x_0_0.set_sampling_freq(self.samp_rate)
        self.analog_sig_source_x_0.set_sampling_freq(self.samp_rate)

    def get_bandwidth(self):
        return self.bandwidth

    def set_bandwidth(self, bandwidth):
        self.bandwidth = bandwidth
        self.set_spreading_ratio(self.samp_rate/(self.bandwidth/1000000))

    def get_spreading_ratio(self):
        return self.spreading_ratio

    def set_spreading_ratio(self, spreading_ratio):
        self.spreading_ratio = spreading_ratio
        self.analog_sig_source_x_0_2.set_frequency((self.cf4-self.offset)*self.spreading_ratio)
        self.analog_sig_source_x_0_1.set_frequency((self.cf3-self.offset)*self.spreading_ratio)
        self.analog_sig_source_x_0_0.set_frequency((self.cf2-self.offset)*self.spreading_ratio)
        self.analog_sig_source_x_0.set_frequency((self.cf1-self.offset)*self.spreading_ratio)

    def get_offset(self):
        return self.offset

    def set_offset(self, offset):
        self.offset = offset
        self.analog_sig_source_x_0_2.set_frequency((self.cf4-self.offset)*self.spreading_ratio)
        self.analog_sig_source_x_0_1.set_frequency((self.cf3-self.offset)*self.spreading_ratio)
        self.analog_sig_source_x_0_0.set_frequency((self.cf2-self.offset)*self.spreading_ratio)
        self.analog_sig_source_x_0.set_frequency((self.cf1-self.offset)*self.spreading_ratio)

    def get_center_freq(self):
        return self.center_freq

    def set_center_freq(self, center_freq):
        self.center_freq = center_freq


def argument_parser():
    parser = OptionParser(usage="%prog: [options]", option_class=eng_option)
    parser.add_option(
        "", "--PSKMod1", dest="PSKMod1", type="intx", default=2,
        help="Set modulation [default=%default]")
    parser.add_option(
        "", "--PSKMod2", dest="PSKMod2", type="intx", default=2,
        help="Set modulation [default=%default]")
    parser.add_option(
        "", "--QAMMod3", dest="QAMMod3", type="intx", default=64,
        help="Set modulation [default=%default]")
    parser.add_option(
        "", "--QAMMod4", dest="QAMMod4", type="intx", default=64,
        help="Set modulation [default=%default]")
    parser.add_option(
        "", "--amplitude1", dest="amplitude1", type="eng_float", default=eng_notation.num_to_str(20),
        help="Set amplitude [default=%default]")
    parser.add_option(
        "", "--amplitude2", dest="amplitude2", type="eng_float", default=eng_notation.num_to_str(20),
        help="Set amplitude [default=%default]")
    parser.add_option(
        "", "--amplitude3", dest="amplitude3", type="eng_float", default=eng_notation.num_to_str(20),
        help="Set amplitude [default=%default]")
    parser.add_option(
        "", "--amplitude4", dest="amplitude4", type="eng_float", default=eng_notation.num_to_str(20),
        help="Set amplitude [default=%default]")
    parser.add_option(
        "", "--bw1", dest="bw1", type="intx", default=2,
        help="Set bw [default=%default]")
    parser.add_option(
        "", "--bw2", dest="bw2", type="intx", default=2,
        help="Set bw [default=%default]")
    parser.add_option(
        "", "--bw3", dest="bw3", type="intx", default=10,
        help="Set bw [default=%default]")
    parser.add_option(
        "", "--bw4", dest="bw4", type="intx", default=10,
        help="Set bw [default=%default]")
    parser.add_option(
        "", "--cf1", dest="cf1", type="intx", default=920,
        help="Set cf [default=%default]")
    parser.add_option(
        "", "--cf2", dest="cf2", type="intx", default=920,
        help="Set cf [default=%default]")
    parser.add_option(
        "", "--cf3", dest="cf3", type="intx", default=920,
        help="Set cf [default=%default]")
    parser.add_option(
        "", "--cf4", dest="cf4", type="intx", default=920,
        help="Set cf [default=%default]")
    parser.add_option(
        "", "--file-name", dest="file_name", type="string", default='default_file.dat',
        help="Set file_name [default=%default]")
    parser.add_option(
        "", "--noiselevel", dest="noiselevel", type="eng_float", default=eng_notation.num_to_str(.01),
        help="Set noiselevel [default=%default]")
    parser.add_option(
        "", "--on-off1", dest="on_off1", type="intx", default=0,
        help="Set on_off [default=%default]")
    parser.add_option(
        "", "--on-off2", dest="on_off2", type="intx", default=0,
        help="Set on_off [default=%default]")
    parser.add_option(
        "", "--on-off3", dest="on_off3", type="intx", default=0,
        help="Set on_off [default=%default]")
    parser.add_option(
        "", "--on-off4", dest="on_off4", type="intx", default=0,
        help="Set on_off [default=%default]")
    return parser


def main(top_block_cls=channel_generator, options=None):
    if options is None:
        options, _ = argument_parser().parse_args()

    tb = top_block_cls(PSKMod1=options.PSKMod1, PSKMod2=options.PSKMod2, QAMMod3=options.QAMMod3, QAMMod4=options.QAMMod4, amplitude1=options.amplitude1, amplitude2=options.amplitude2, amplitude3=options.amplitude3, amplitude4=options.amplitude4, bw1=options.bw1, bw2=options.bw2, bw3=options.bw3, bw4=options.bw4, cf1=options.cf1, cf2=options.cf2, cf3=options.cf3, cf4=options.cf4, file_name=options.file_name, noiselevel=options.noiselevel, on_off1=options.on_off1, on_off2=options.on_off2, on_off3=options.on_off3, on_off4=options.on_off4)
    tb.start()
    try:
        raw_input('Press Enter to quit: ')
    except EOFError:
        pass
    tb.stop()
    tb.wait()


if __name__ == '__main__':
    main()
