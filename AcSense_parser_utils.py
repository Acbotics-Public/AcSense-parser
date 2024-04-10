import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from time import sleep
import os
import glob
import pandas as pd
from pynmeagps import NMEAReader
import datetime

##############################
# AcSense parser utilities for use with
# AcSense-Mini
# AcSense-Mini-48hr
# AcSense-8Channel
# AcSense-8Channel-PLUS

##############################
# config
##############################
BLOCK_SIZE = 512
FS_AUDIO = 52734  # change to reflect your config file!
TICK = 10e-9  # sample interval is s


##############################
# Common unitities:
# read timestamp
def read_timestamp64(f):
    return int.from_bytes(f.read(8), "little")


# Load and process all sens and acoustic files:
def load_process_all_files(
    indir, outdir, hydrophone_ADC="EXT", plotting=False, export=True, audiofile=True
):
    # indir: input directory (search for SENS*.dat here)
    # outdir: output directory (where to put plots and csvs)
    # load and process all files indicated by root directory indir.
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    SENS_filelist = sorted(glob.glob(os.path.join(indir, "SENS*.dat")))
    AUDIO_filelist = sorted(glob.glob(os.path.join(indir, "AC*.dat")))
    if len(SENS_filelist) == 0:
        # go one more level:
        SENS_filelist = sorted(
            glob.glob(os.path.join(os.path.join(indir, "*"), "SENS*.dat"))
        )
        AUDIO_filelist = sorted(
            glob.glob(os.path.join(os.path.join(indir, "*"), "AC*.dat"))
        )

    for filen in SENS_filelist:
        print("Working on file " + filen)
        # parse and save to csv:
        export_csv_sens_data(
            sens_filename=filen,
            outdir=outdir,
            plotting=plotting,
            export=export,
            hydrophone_ADC=hydrophone_ADC,
        )

    # export audio files:
    if audiofile:
        for filen in AUDIO_filelist:
            print("Working on file " + filen)
            export_csv_audio(
                audio_filename=filen,
                outdir=outdir,
                plotting=plotting,
                hydrophone_ADC=hydrophone_ADC,
            )


###############################
# ADC file utilities:
###############################
# Parse recorded audio file:
def parse_record_audio(f, hydrophone_ADC="EXT"):
    res3 = {}
    if hydrophone_ADC == "INT":  # if internal adc
        res3["versionId"] = int.from_bytes(f.read(1), "little")

        res3["channels"] = int.from_bytes(f.read(1), "little")
        res3["bitsPerChannel"] = int.from_bytes(f.read(1), "little")
        res3["bytesPerChannel"] = int.from_bytes(f.read(1), "little")
        res3["unpackedShiftRight"] = int.from_bytes(f.read(1), "little")
        res3["overflowCount"] = int.from_bytes(f.read(1), "little")
        res3["dataRecordsPerBuffer"] = int.from_bytes(f.read(2), "little")
        res3["sampleRate"] = np.frombuffer(f.read(8), dtype=np.float64)[0]
        res3["sampleCount"] = int.from_bytes(f.read(4), "little")

        res3["timestamp"] = read_timestamp64(f)

        res3["scale"] = np.frombuffer(f.read(8), dtype=np.float64)[0]
        tt = []

        print(res3)
        if res3["channels"] == 255:
            raise Exception("Something is wrong!")
        if res3["channels"] > 1:
            data1 = [[] for i in range(res3["channels"])]
        else:
            data1 = []

        # print(str(len(tt)) + " " + str(len(data1)))
        # print(res3["channels"])
        # blarg
        for i in range(res3["dataRecordsPerBuffer"]):

            tt.append(res3["timestamp"] + i * 1 / (FS_AUDIO * TICK))

            if res3["channels"] > 1:
                for ch in range(res3["channels"]):
                    d = int.from_bytes(f.read(2), "little", signed=True)
                    data1[ch].append(d)
            else:
                d = int.from_bytes(f.read(2), "little", signed=True)
                # print(d)
                data1.append(d)
            # print(res3["channels"])
        print(str(len(tt)) + " " + str(len(data1)))
        # blarg
        res3["data"] = data1
        res3["time_vector"] = tt

    elif hydrophone_ADC == "EXT":
        res3["timestamp"] = read_timestamp64(f)
        res3["dataRecordsPerBuffer"] = int.from_bytes(f.read(2), "little")
        res3["channels"] = int.from_bytes(f.read(1), "little")
        res3["bytesPerChannel"] = int.from_bytes(f.read(1), "little")
        res3["overFlowCount"] = int.from_bytes(f.read(2), "little")
        res3["firstOverFlowRecord"] = int.from_bytes(f.read(2), "little")
        res3["time_vector"] = []
        data1 = [[] for i in range(res3["channels"])]
        for i in range(255):
            res3["time_vector"].append(res3["timestamp"] + i * 1 / (FS_AUDIO * TICK))

            for ch in range(res3["channels"]):
                data1[ch].append(int.from_bytes(f.read(2), "little", signed=True))
        res3["data"] = data1
    return res3


def load_raw_audio_data(filename, hydrophone_ADC="EXT"):
    # input: filename to audio file
    # output: raw data from audio file

    f = open(filename, "rb")
    f.seek(0x00)
    if hydrophone_ADC == "INT":
        raw_data = []
        seek_offset = 506  # 524
    elif hydrophone_ADC == "EXT":
        raw_data = [[] for i in range(8)]
        seek_offset = 0x1000
    time_stamps = []
    while True:
        block_start = f.tell()
        try:
            data = parse_record_audio(f, hydrophone_ADC)
        except:
            return [None, None, None]
        time_stamps.extend(data["time_vector"])
        if hydrophone_ADC == "EXT":
            for i in range(data["channels"]):
                raw_data[i].extend(data["data"][i])
        else:
            raw_data.extend(data["data"])
        f.seek(block_start + seek_offset)
        if f.tell() >= os.fstat(f.fileno()).st_size:
            break
    return [np.array(time_stamps), np.array(raw_data), data["channels"]]


def export_csv_audio(audio_filename, outdir, plotting=False, hydrophone_ADC="EXT"):
    # INput:
    #    sens_filename: filename of sens file parsed
    #    outdir (optional): where to put .csv files from the parse
    [timestamps, raw_data, nch] = load_raw_audio_data(audio_filename, hydrophone_ADC)
    if timestamps is None:
        return
    fileroot = (os.path.split(audio_filename))[-1].split(".")[0]
    # print(fileroot)
    print(raw_data.shape)
    print(timestamps.shape)
    outroot = os.path.join(outdir, fileroot)
    plt.close("all")
    plt.subplot(211)
    plt.plot(timestamps, raw_data[0 : len(timestamps)])
    plt.subplot(212)
    plt.specgram(raw_data + 1, Fs=52000, NFFT=2000)
    plt.savefig(outroot + ".png")

    # from sens dict, export csvs for each:
    # plt.show()
    df_Audio = pd.DataFrame(
        data=np.transpose(raw_data), columns=list(range(0, nch)), index=timestamps
    )

    df_Audio.to_csv(outroot + "_Audio.csv")
    return


################################
# SENS file utilities:
def read_int_adc_data(f):
    res = {}
    res["timestamp"] = read_timestamp64(f)
    adc_data = []
    for i in range(3):
        adc_data.append(int.from_bytes(f.read(2), "little", signed=True))
    res["adc_data"] = adc_data
    return res


def read_index(f):  # Used in SENS files
    return int.from_bytes(f.read(2), "little")


def read_block_header(f):
    num_entries = int.from_bytes(f.read(2), "little")
    fill_index = int.from_bytes(f.read(2), "little")
    size_fill = int.from_bytes(f.read(2), "little")
    return {
        "num_entries": num_entries,
        "fill_index": fill_index,
        "size_fill": size_fill,
    }


def read_int_pts_data(f):
    timestamp = read_timestamp64(f)
    pressure_mbarX100 = int.from_bytes(f.read(4), "little")
    TempCdegsX100 = int.from_bytes(f.read(4), "little", signed=True)
    return {
        "timestamp": timestamp,
        "pressure_mbarX100": pressure_mbarX100,
        "TempCdegsX100": TempCdegsX100,
    }


def read_external_pts_data(f):
    timestamp = read_timestamp64(f)
    pressure_barX100 = int.from_bytes(f.read(4), "little", signed=True)
    TempCdegsX100 = int.from_bytes(f.read(4), "little", signed=True)
    # print("in exteranl pressure read")
    return {
        "timestamp": timestamp,
        "pressure_barX100": pressure_barX100,
        "TempCdegsX100": TempCdegsX100,
    }


def read_rtc_data(f):
    timestamp = read_timestamp64(f)
    id1 = int.from_bytes(f.read(1), "little")
    id2 = int.from_bytes(f.read(1), "little")
    nbytes = int.from_bytes(f.read(2), "little")
    sec_bcd = int.from_bytes(f.read(1), "little")
    min_bcd = int.from_bytes(f.read(1), "little")
    hour_bcd = int.from_bytes(f.read(1), "little")
    day = int.from_bytes(f.read(1), "little")
    date_bcd = int.from_bytes(f.read(1), "little")
    mon_bcd = int.from_bytes(f.read(1), "little")
    year_bcd = int.from_bytes(f.read(1), "little")
    res = {}
    res["timestamp"] = timestamp

    res["seconds"] = (sec_bcd & 0xF) + (((sec_bcd & 0xF0) >> 4) * 10)
    res["minutes"] = (min_bcd & 0xF) + (((min_bcd & 0xF0) >> 4) * 10)
    res["hours"] = (hour_bcd & 0xF) + (((hour_bcd & 0x30) >> 4) * 10)
    res["wday"] = day
    res["mday"] = (date_bcd & 0xF) + (((date_bcd & 0x30) >> 4) * 10)
    res["month"] = (mon_bcd & 0xF) + (((mon_bcd & 0x10) >> 4) * 10)
    res["year"] = (year_bcd & 0xF) + (((year_bcd & 0xF0) >> 4) * 10)
    res["timestr"] = "{0:04d}{1:02d}{2:02d}T{3:02d}{4:02d}{5:02d}".format(
        2000 + res["year"],
        res["month"],
        res["mday"],
        res["hours"],
        res["minutes"],
        res["seconds"],
    )
    # print(res['timestr'])
    # blarg
    # print(hex(year_bcd))
    return res


def read_imu_data(f):
    # read imu data, get metrics
    res = {}
    res["timestamp"] = read_timestamp64(f)
    res["ID1"] = int.from_bytes(f.read(1), "little", signed=False)
    res["ID2"] = int.from_bytes(f.read(1), "little", signed=False)
    res["numbytes"] = int.from_bytes(f.read(2), "little", signed=False)
    res["PitchNed_DegreesX100"] = int.from_bytes(f.read(4), "little", signed=True)
    res["RollNed_DegreesX100"] = int.from_bytes(f.read(4), "little", signed=True)
    res["Accel_X"] = int.from_bytes(f.read(2), "little", signed=True)
    res["Accel_Y"] = int.from_bytes(f.read(2), "little", signed=True)
    res["Accel_Z"] = int.from_bytes(f.read(2), "little", signed=True)
    res["Gyro_X"] = int.from_bytes(f.read(2), "little", signed=True)
    res["Gyro_Y"] = int.from_bytes(f.read(2), "little", signed=True)
    res["Gyro_Z"] = int.from_bytes(f.read(2), "little", signed=True)
    return res


def read_generic_data(f):
    # read generic binary data from binary
    timestamp = read_timestamp64(f)
    id1 = int.from_bytes(f.read(1), "little", signed=True)
    id2 = int.from_bytes(f.read(1), "little", signed=True)
    bts = int.from_bytes(f.read(2), "little", signed=True)

    if id1 == ord("A"):
        data = []
        for i in range(int(bts / 2)):
            data.append(int.from_bytes(f.read(2), "little", signed=True))

    elif id1 == ord("E"):
        # print("BAR30")
        pressure_barX100 = int.from_bytes(f.read(4), "little", signed=True)
        TempCdegsX100 = int.from_bytes(f.read(4), "little", signed=True)
        data = {
            "timestamp": timestamp,
            "pressure_barX100": pressure_barX100,
            "TempCdegsX100": TempCdegsX100,
        }

    return {"timestamp": timestamp, "data": data, "id1": id1, "id2": id2}


def read_string_data(f):
    # read string data from binary
    timestamp = read_timestamp64(f)
    id1 = int.from_bytes(f.read(1), "little", signed=True)
    id2 = int.from_bytes(f.read(1), "little", signed=True)
    bts = int.from_bytes(f.read(2), "little", signed=True)

    st = f.read(bts).decode().strip("\x00")
    return {"timestamp": timestamp, "str": st}


def read_ping_data(f):
    res = {}
    res["timestamp"] = read_timestamp64(f)
    res["distance"] = int.from_bytes(f.read(4), "little")
    res["confidence"] = int.from_bytes(f.read(2), "little")
    res["transmit_duration"] = int.from_bytes(f.read(2), "little")
    res["ping_number"] = int.from_bytes(f.read(4), "little")
    res["scan_start"] = int.from_bytes(f.read(4), "little")
    res["scan_length"] = int.from_bytes(f.read(4), "little")
    res["gain_setting"] = int.from_bytes(f.read(4), "little")
    res["profile_data_length"] = int.from_bytes(f.read(2), "little")
    res["profile_values"] = [int(x) for x in f.read(200)]
    return res


def parse_record(f, sens_dict, hydrophone_ADC="EXT", timeonly=False):
    # hydrophone_ADC = "EXT" if using 8ch or 16ch, "INT" if a mini system
    next_index = read_index(f)
    msg_id = int.from_bytes(f.read(2), "little")
    int_adc = None
    first_idx = 0
    temp_dict = {}
    if msg_id == 0xF:  # internal ADC
        if not timeonly:
            IntADCRead(f, sens_dict, hydrophone_ADC)
        else:
            IntADCRead(f, temp_dict, hydrophone_ADC)
            first_idx = np.min(temp_dict["timestamp"])

    elif msg_id == 0xB:  # internal PTS
        if not timeonly:
            IntPTSRead(f, sens_dict)
        else:
            IntPTSRead(f, {})

        # print('pts_data= ' + repr(pts_data))
    elif msg_id == 0xC:  # imu
        if not timeonly:
            IMURead(f, sens_dict)
        else:
            IMURead(f, temp_dict)
            first_idx = np.min(temp_dict["timestamp"])
        # print("imu_data= " + repr(imur_data))

    elif msg_id == 0xE:  # RTC
        RTCRead(f, sens_dict)

        # print('rtc_data: ' + repr(rtc_data))
    elif msg_id == 0x10:  # External Pressure/temp
        if not timeonly:
            ExtPTSRead(f, sens_dict)
        else:
            ExtPTSRead(f, {})
        # print('external_pts: ' + repr(pts_data))
    elif msg_id == 0x12:  # Generic Data
        gen_data = read_generic_data(f)
        if not timeonly:
            if gen_data["id1"] == ord("A"):
                IntADCRead2(sens_dict, gen_data)
            elif gen_data["id1"] == ord("E"):
                ExtPTSRead2(sens_dict, gen_data)
            else:
                pass
        #   print("UNKNOWN GENERIC ID")
    elif msg_id == 0x11:  # string
        str_data = GPSRead(f)
        # str_data['EpochDate']=msg.date
        if str_data is not None:
            if "GPS_data" in sens_dict.keys():
                sens_dict["GPS_data"].append(str_data)
            else:
                sens_dict["GPS_data"] = [str_data]

        # print(str_data)
    else:
        pass
        # print("****MSG_ID: " + hex(msg_id))

    return [{"next_index": next_index}, sens_dict]


def GPSRead(f):
    str_data = read_string_data(f)
    msg = NMEAReader.parse(str_data["str"])
    if "RMC" in str_data["str"] and "$" in str_data["str"]:
        str_data["EpochTime"] = msg.time
        str_data["EpochDate"] = msg.date
        str_data["Lat"] = msg.lat
        str_data["Lon"] = msg.lon
        str_data["timestr"] = (
            str(str_data["EpochDate"]) + "_T" + str(str_data["EpochTime"])
        )
        try:
            str_data["UnixTime"] = datetime.datetime.strptime(
                str_data["timestr"] + "Z", "%Y-%m-%d_T%H:%M:%S.%f%z"
            ).timestamp()
        except:
            str_data["UnixTime"] = datetime.datetime.strptime(
                str_data["timestr"] + "Z", "%Y-%m-%d_T%H:%M:%S%z"
            ).timestamp()

        return str_data
    elif "$" in str_data["str"]:
        return None
        # print(msg)
        try:
            str_data["EpochTime"] = msg.time
        except:
            str_data["EpochTime"] = ""
        try:
            str_data["Lat"] = msg.lat
        except:
            str_data["Lat"] = ""
        try:
            str_data["Lon"] = msg.lat
        except:
            str_data["Lon"] = ""
        try:
            str_data["EpochDate"] = msg.date
        except:
            str_data["EpochDate"] = ""
        try:
            str_data["timestr"] = (
                str(str_data["EpochDate"]) + "_T" + str(str_data["EpochTime"])
            )
        except:
            str_data["timestr"] = ""


def ExtPTSRead2(sens_dict, gen_data):
    pts_data = gen_data["data"]
    if "ExtPTS_data" in sens_dict.keys():
        sens_dict["ExtPTS_data"].append(pts_data)
    else:
        sens_dict["ExtPTS_data"] = [pts_data]


def IntADCRead2(sens_dict, gen_data):
    adc_timestamp = gen_data["timestamp"]
    int_adc = gen_data["data"]
    dict_adc = {"timestamp": adc_timestamp, "adc_data": int_adc}

    if "Int_ADC" in sens_dict.keys():
        sens_dict["Int_ADC"].append(dict_adc)
    else:
        sens_dict["Int_ADC"] = [dict_adc]


def ExtPTSRead(f, sens_dict):
    pts_data = read_external_pts_data(f)
    # print(pts_data)
    if "ExtPTS_data" in sens_dict.keys():
        sens_dict["ExtPTS_data"].append(pts_data)
    else:
        sens_dict["ExtPTS_data"] = [pts_data]


def RTCRead(f, sens_dict):
    rtc_data = read_rtc_data(f)
    if "RTC_data" in sens_dict.keys():
        sens_dict["RTC_data"].append(rtc_data)
    else:
        sens_dict["RTC_data"] = [rtc_data]


def IMURead(f, sens_dict):
    imu_data = read_imu_data(f)
    if "IMU_data" in sens_dict.keys():
        sens_dict["IMU_data"].append(imu_data)
    else:
        sens_dict["IMU_data"] = [imu_data]


def IntPTSRead(f, sens_dict):
    pts_data = read_int_pts_data(f)
    if "IntPTS_data" in sens_dict.keys():
        sens_dict["IntPTS_data"].append(pts_data)
    else:
        sens_dict["IntPTS_data"] = [pts_data]


def IntADCRead(f, sens_dict, hydrophone_ADC):
    adc_data = read_int_adc_data(f)
    if hydrophone_ADC == "EXT":  # assume just reading ADC plain!
        used_adc_data = adc_data
    else:
        used_adc_data = {
            "timestamp": adc_data["timestamp"],
            "adc_data": adc_data["adc_data"][0],
        }
        # repackage, bc only using 1 of the entries?
    if "Int_ADC" in sens_dict.keys():
        sens_dict["Int_ADC"].append(used_adc_data)
    else:
        sens_dict["Int_ADC"] = [used_adc_data]


def write_and_plot_sens(
    outdir, outroot, data, plotting=False, export=True, hydrophone_ADC="EXT"
):
    if plotting:
        plot_sens_data(data, outdir, outroot, hydrophone_ADC)

    if export:
        if "Int_ADC" in data.keys():
            ADC_data = data["Int_ADC"]
            tstamp = np.array([A["timestamp"] for A in ADC_data])
            if hydrophone_ADC == "EXT":
                V_geophone = np.array([A["adc_data"][0] for A in ADC_data])
                L_geophone = np.array([A["adc_data"][1] for A in ADC_data])
                T_geophone = np.array([A["adc_data"][2] for A in ADC_data])
                df_sens_ADC = pd.DataFrame(
                    {
                        "timestamp": tstamp,
                        "V_geophone": V_geophone,
                        "L_geophone": L_geophone,
                        "T_geophone": T_geophone,
                    }
                )
                df_sens_ADC.to_csv(os.path.join(outdir, outroot + "_ADC_geophone.csv"))

            else:  # NOTE: Later, adjust to have options for multiple hydrophones on internal ADC as needed
                hydrophone = np.array([A["adc_data"][0] for A in ADC_data])
                xsamp = np.linspace(
                    0, len(hydrophone), num=len(hydrophone), endpoint=False
                )
                x_tstamp = np.linspace(
                    0, len(hydrophone), num=len(tstamp), endpoint=False
                )
                timestamps_all = np.interp(xsamp, x_tstamp, tstamp)
                df_sens_ADC = pd.DataFrame(
                    {"timestamp": timestamps_all, "Hydrophone": hydrophone}
                )

                df_sens_ADC.to_csv(
                    os.path.join(outdir, outroot + "_ADC_hydrophone.csv")
                )

        if "IntPTS_data" in data.keys():
            print("Exporting internal Pressure/temp data...")

            df_sens_internal_PTS = pd.DataFrame(data["IntPTS_data"])
            df_sens_internal_PTS.to_csv(
                os.path.join(outdir, outroot + "_internal_PTS.csv")
            )

        if "ExtPTS_data" in data.keys():
            print("Exporting external Pressure/temp data...")
            df_sens_external_PTS = pd.DataFrame(data["ExtPTS_data"])
            df_sens_external_PTS.to_csv(
                os.path.join(outdir, outroot + "_external_PTS.csv")
            )

        if "IMU_data" in data.keys():
            print("Exporting IMU data...")
            df_sens_IMU = pd.DataFrame(data["IMU_data"])
            df_sens_IMU.to_csv(os.path.join(outdir, outroot + "_IMU.csv"))

        if "GPS_data" in data.keys():
            print("Exporting GPS data...")
            df_sens_GPS = pd.DataFrame(data["GPS_data"])
            df_sens_GPS.to_csv(os.path.join(outdir, outroot + "_GPS.csv"))
        if "RTC_data" in data.keys():
            df_sens_RTC = pd.DataFrame(data["RTC_data"])
            df_sens_RTC.to_csv(os.path.join(outdir, outroot + "_RTC.csv"))
        if "PING_data" in data.keys():
            df_sens_PING = pd.DataFrame(data["PING_data"])
            df_sens_PING.to_csv(os.path.join(outdir, outroot + "_Ping_echosounder.csv"))


def read_all_timedata(sens_filename, hydrophone_ADC="EXT"):
    # input: full path to filename with SENS data

    f, readchunks, idx, split_count = prep_read(sens_filename)
    sens_dict = {}
    print("Loading data...")
    while True:
        sens_dict = read_block(hydrophone_ADC, f, sens_dict, timeonly=True)
        idx = idx + 1
        if f.tell() >= os.fstat(f.fileno()).st_size:
            break
    return sens_dict


def export_csv_sens_data(
    sens_filename, outdir, plotting, hydrophone_ADC="EXT", export=True
):
    # INput:
    #    sens_filename: filename of sens file parsed
    #    outdir (optional): where to put .csv files from the parse
    # mini: if true, pull acoustics!

    fileroot = os.path.split(sens_filename)[-1].split(".")[0]

    # input: full path to filename with SENS data

    f, readchunks, idx, split_count = prep_read(sens_filename)
    sens_dict = {}
    print("Loading data...")
    while True:
        sens_dict = read_block(hydrophone_ADC, f, sens_dict)
        idx = idx + 1
        if f.tell() >= os.fstat(f.fileno()).st_size:
            break
        if idx >= readchunks:
            print("Reached max size for one csv set, plotting and writing...")

            outroot = fileroot + "_n" + str(split_count)

            write_and_plot_sens(
                outdir=outdir,
                outroot=outroot,
                data=sens_dict,
                plotting=plotting,
                export=export,
                hydrophone_ADC=hydrophone_ADC,
            )
            split_count = split_count + 1
            idx = 0
            sens_dict = {}
            print("Saving, RESTARTING DICT!")
    outroot = fileroot + "_n" + str(split_count)
    print("Finishing up with final plot and write...")
    write_and_plot_sens(
        outdir=outdir,
        outroot=outroot,
        data=sens_dict,
        plotting=plotting,
        export=export,
        hydrophone_ADC=hydrophone_ADC,
    )
    print("DONE!")
    print("Your files will be in " + outdir)


def prep_read(sens_filename):
    f = open(sens_filename, "rb")
    readchunks = 100000  # readchunks
    idx = 0
    split_count = 0
    f.seek(0x00)
    return f, readchunks, idx, split_count


def read_block(hydrophone_ADC, f, sens_dict, timeonly=False):
    block_start = f.tell()
    header = read_block_header(f)
    already_told = 0
    # print('header: ' + repr(header))

    for i in range(header["num_entries"]):
        data, sens_dict = parse_record(
            f, sens_dict, hydrophone_ADC=hydrophone_ADC, timeonly=timeonly
        )
        next_index = data["next_index"] + block_start
        if next_index < f.tell() and not already_told:
            print("Seeking backwards. Issue with parse?" + repr(next_index - f.tell()))
            break
            already_told = 1
        f.seek(next_index)
    curr_pos = f.tell()
    # start on next block
    offset = BLOCK_SIZE - (curr_pos % BLOCK_SIZE)
    offset = offset % BLOCK_SIZE
    next_block_start = curr_pos + offset
    f.seek(next_block_start)

    return sens_dict


################################
# Plotting Utilities
#################################
def plot_sens_data(data, outdir, fileroot, hydrophone_ADC):
    # plot the data!
    plt_len = 0
    data_keys_plot = [
        "GPS_data",
        "RTC_data",
        "Int_ADC",
        "IMU_data",
        "ExtPTS_data",
        "IntPTS_data",
    ]
    for key in data_keys_plot:
        if key in data.keys():
            plt_len = plt_len + 1
    fig, ax = plt.subplots(plt_len, 1, figsize=[20, 10])
    curidx = 0
    # first: plot IMU:
    if "GPS_data" in data.keys():
        gps_RTC = data[
            "GPS_data"
        ]  # [[x['EpochTime'] for x in data['GPS_data']],'Time UTC']
        plot_lat_lon(data["GPS_data"], ax[curidx])
        curidx = curidx + 1
        timescale = gps_RTC
    else:
        gps_RTC = None
        timescale = None
    if "RTC_data" in data.keys():
        # set up x-axis for rtc clock instead:
        timescale = data["RTC_data"]

    # print(gps_RTC)
    if "Int_ADC" in data.keys():
        if hydrophone_ADC == "INT":
            plot_adc_mini(
                get_xaxis(data["Int_ADC"], timescale), data["Int_ADC"], ax[curidx]
            )
        else:
            plot_geophone(
                get_xaxis(data["Int_ADC"], timescale), data["Int_ADC"], ax[curidx]
            )
        curidx = curidx + 1
    if "IMU_data" in data.keys():
        plot_IMU(get_xaxis(data["IMU_data"], timescale), data["IMU_data"], ax[curidx])
        curidx = curidx + 1

    # plot_ping(get_xaxis(data['PING_data'],gps_RTC),data['PING_data'],ax[0,1])
    if "ExtPTS_data" in data.keys():
        plot_PTS(
            get_xaxis(data["ExtPTS_data"], timescale),
            data["ExtPTS_data"],
            "ext",
            ax[curidx],
        )
        curidx = curidx + 1
    if "IntPTS_data" in data.keys():
        plot_PTS(
            get_xaxis(data["IntPTS_data"], timescale),
            data["IntPTS_data"],
            "int",
            ax[curidx],
        )
        curidx = curidx + 1

    plt.tight_layout()
    #    plt.show()
    plt.savefig(os.path.join(outdir, fileroot + "_sens.png"))
    # plt.show()
    plt.close("all")


def get_xaxis(sensor_data, RTC_data=None):
    tstamps = np.array([x["timestamp"] for x in sensor_data]) * TICK
    tstamps = tstamps - tstamps[0]

    if RTC_data is not None:
        # use RTC data to look up timestamps:
        # print(RTC_data[0])
        if "timestr" in RTC_data[0].keys():
            rtc_start = RTC_data[0]["timestr"]
            rtc_end = RTC_data[-1]["timestr"]
            xlabeln = "s since " + rtc_start
        else:
            xlabeln = "s"
    else:
        xlabeln = "s"

    return [tstamps, xlabeln]


def plot_adc_mini(xaxis, ADC_data, ax):
    x_axis = xaxis[0]
    # print(len(x_axis))
    xlabel = xaxis[1]
    hydrophone = np.array([A["adc_data"] for A in ADC_data]).flatten()
    # print(len(hydrophone))
    tstamp = np.array([A["timestamp"] for A in ADC_data]).flatten()
    xsamp = np.linspace(0, len(hydrophone), num=len(hydrophone), endpoint=False)
    x_tstamp = np.linspace(0, len(hydrophone), num=len(tstamp), endpoint=False)
    timestamps_all = np.interp(xsamp, x_tstamp, tstamp) * TICK
    ax.plot(timestamps_all, hydrophone, ".")
    ax.set_title("Hydrophone (ADC data)")
    ax.set_xlabel(xlabel)


def plot_specadc_mini(xaxis, ADC_data, ax):
    x_axis = xaxis[0]

    xlabel = xaxis[1]
    hydrophone = np.array([A["adc_data"] for A in ADC_data]).flatten()

    ax.specgram(hydrophone, Fs=FS_AUDIO, NFFT=5000, noverlap=4000)
    # ax.set_ylim([0,5000])
    ax.set_title("Hydrophone (ADC data)")
    ax.set_xlabel(xlabel)


def plot_PTS(xaxis, PTS_data, ptype, ax):
    x_axis = xaxis[0]
    xlabel = xaxis[1]
    if ptype == "int":
        pvar = "pressure_mbarX100"
        ax.set_title("Internal P/T")
    else:
        pvar = "pressure_barX100"
        ax.set_title("External P/T")
    # plot PTS data!
    ax.plot(x_axis, [x[pvar] / 100 for x in PTS_data], "g.")
    ax.yaxis.label.set_color("green")
    ax.set_ylabel("Pressure (bar)")
    ax2 = ax.twinx()
    ax2.plot(x_axis, [x["TempCdegsX100"] / 100 for x in PTS_data], "b.")
    ax2.set_ylabel("Temp (deg C)")
    ax2.yaxis.label.set_color("blue")
    ax.set_xlabel(xlabel)


def plot_ping(xaxis, PING_data, ax):
    x_axis = xaxis[0]
    xlabel = xaxis[1]
    ping_profile_vals = np.array([np.array(x["profile_values"]) for x in PING_data])

    ranges = np.array(list(range(0, len(PING_data[0]["profile_values"]))))
    range_in_m = ranges / max(ranges) * 24 + 1

    xmat, ymat = np.meshgrid(range(0, len(x_axis)), range_in_m)

    ax.pcolormesh(xmat, ymat, np.log10((np.transpose(ping_profile_vals) + 1) / 256))
    ax.set_xlabel("time bin, " + xlabel)
    ax.set_ylabel("Range bin")


def plot_geophone(xaxis, ADC_data, ax):
    # plot geophone data
    x_axis = xaxis[0]

    xlabel = xaxis[1]
    V_geophone = np.array([A["adc_data"][0] for A in ADC_data])
    L_geophone = np.array([A["adc_data"][1] for A in ADC_data])
    T_geophone = np.array([A["adc_data"][2] for A in ADC_data])

    ax.plot(x_axis, V_geophone, ".")
    ax.plot(x_axis, L_geophone, ".")
    ax.plot(x_axis, T_geophone, ".")
    ax.set_title("Geophones (ADC data)")
    ax.legend(["V", "L", "T"])
    ax.set_xlabel(xlabel)


def plot_IMU(xaxis, IMU_data, ax):
    x_axis = xaxis[0]
    xlabel = xaxis[1]

    ax.plot(x_axis, [x["PitchNed_DegreesX100"] / 100 for x in IMU_data])
    ax.plot(x_axis, [x["RollNed_DegreesX100"] / 100 for x in IMU_data])
    ax.legend(["Pitch NED deg", "Roll NED deg"])
    ax.set_ylabel("Degrees")
    ax.set_title("IMU Pitch/roll")
    ax.set_xlabel(xlabel)


def plot_lat_lon(gps_data, ax):
    ax.clear()
    lat_vec = []
    lon_vec = []
    for entry in gps_data:
        if "RMC" in entry["str"]:
            if "Lon" in entry.keys():
                if entry["Lon"] != "":
                    lon_vec.append(entry["Lon"])
                    lat_vec.append(entry["Lat"])
    ax.plot(lon_vec, lat_vec)
    ax.set_xlabel("Lon")
    ax.set_ylabel("Lat")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
