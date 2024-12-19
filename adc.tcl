# This moduel contains the interface for controlling the 4 slow ADC-ports
# on RedPitaya. With this they are accesible via AXI-lite from
# PS by polling regs 0x00 to 0x0C

# Added fast rf dac to interface:

#source parameters.tcl from build_src
source ../../../build_src/fpga_tcl/parameters.tcl

#cell xilinx.com:ip:xadc_wiz xadc {
#    INTERFACE_SELECTION None
#    XADC_STARUP_SELECTION channel_sequencer
#    ENABLE_AXI4STREAM true
#    DCLK_FREQUENCY 125
#    CHANNEL_ENABLE_VP_VN false
#    CHANNEL_ENABLE_VAUXP0_VAUXN0 true
#    CHANNEL_ENABLE_VAUXP1_VAUXN1 true
#    CHANNEL_ENABLE_VAUXP8_VAUXN8 true
#    CHANNEL_ENABLE_VAUXP9_VAUXN9 true
#    ENABLE_RESET false
#    WAVEFORM_TYPE CONSTANT
#    SEQUENCER_MODE Continuous
#    ADC_CONVERSION_RATE 1000
#    EXTERNAL_MUX_CHANNEL VP_V
#    SINGLE_CHANNEL_SELECTION TEMPERATURE
#    STIMULUS_FREQ 1.0
#}
#
#cell hhi-thz:user:axi_xadc_controller_v1_0 axi_xadc_controller {
#
#    RESET_WIDTH $RESET_WIDTH
#    RESET_INDEX $RESET_INDEX_XADC_CTRL
#
#} {
#
#    s_axis xadc/M_AXIS
#}

# ADC IP Core
cell hhi-thz:user:axi_red_pitaya_adc_v2_0 axi_red_pitaya_adc {
    RESET_WIDTH $RESET_WIDTH
    RESET_INDEX $RESET_INDEX_RP_ADC
    TRIGGER_FROM_SAME_CLOCK 1

} {

}

# Dummy data generator IP core
cell hhi-thz:user:axis_dummy_data_generator_v2_0 dummy_data_gen {
    RESET_WIDTH $RESET_WIDTH
    RESET_INDEX $RESET_INDEX_DUMMY_DATA_GEN
    INT_BRAM_DEPTH $BRAM_AXI_DEPTH
    INT_BRAM_CTRL_ADDR_WIDTH $BRAM_AXI_CTRL_ADDR_WIDTH
    INT_BRAM_ADDR_OFFSET $BRAM_AXI_ADDR_OFFSET
} {
    aclk axi_red_pitaya_adc/aclk
    rstn axi_red_pitaya_adc/rstn
    s_axi_aresetn axi_red_pitaya_adc/s_axi_aresetn
}

# AXI Stream Multiplexer IP core
cell hhi-thz:user:axis_mux_4x1_v1_0 axis_mux_4x1 {
} {
    aclk axi_red_pitaya_adc/aclk
    s_axi_aresetn axi_red_pitaya_adc/s_axi_aresetn
    S_AXIS_0 axi_red_pitaya_adc/M_AXIS
    S_AXIS_1 dummy_data_gen/M_AXIS
}
