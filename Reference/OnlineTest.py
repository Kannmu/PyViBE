import binascii
import socket
import struct
import sys
import time
from urllib import parse
# import mne
import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
import scipy
import serial
import FBCCA
import CCA
plt.style.use("seaborn-v0_8")
plt.rcParams["font.family"] = ["Times New Roman"]
plt.rcParams["axes.labelsize"] = 14

############### Parameters ##############

SampleTime = 3

# SignalMatrix = np.ones((1,2))

AllDataBytes = bytes()

PackageNum = 0

Headers = {}

Predict = -1

FreqList = [7, 8.5, 10, 11.5]

WeightList = [1, 1, 1, 1]

BreakTime = 0.1

SendingList = [2,2,1,1,2]

FBCCA = FBCCA.FBCCA(ws = SampleTime, Fs = 512, Nf = 4, Nc = 1,Nb = 8)
# CCA = CCA.CCA_Base(ws = SampleTime, Fs = 512, Nf = 4, Nc = 1)
# 创建TCP服务器

# signal_socket = 0
# stimu_socket = 0
# Signal_Address = 0

def ConnectOpenViBE():
    global signal_socket
    signal_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # stimu_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 绑定服务器地址和端口
    Signal_Address = ("localhost", 31415)  # 这里的地址可以根据需要修改

    signal_socket.connect(Signal_Address)

def decode_header(byte_data):
    format_version = struct.unpack("<I", byte_data[0:4])[0]
    endianness = struct.unpack("<I", byte_data[4:8])[0]
    sampling_frequency = struct.unpack("<I", byte_data[8:12])[0]
    num_channels = struct.unpack("<I", byte_data[12:16])[0]
    samples_per_chunk = struct.unpack("<I", byte_data[16:20])[0]
    reserved0 = struct.unpack("<I", byte_data[20:24])[0]
    reserved1 = struct.unpack("<I", byte_data[24:28])[0]
    reserved2 = struct.unpack("<I", byte_data[28:32])[0]

    return {
        "format_version": format_version,
        "endianness": endianness,
        "sampling_frequency": sampling_frequency,
        "num_channels": num_channels,
        "samples_per_chunk": samples_per_chunk,
        "reserved0": reserved0,
        "reserved1": reserved1,
        "reserved2": reserved2,
    }

def read_data():
    global signal_socket
    data = signal_socket.recv(2048)
    return data

def decode_data(byte_data, n_channels, n_samples, n_packages):
    global Offset
    data = np.zeros((n_packages * n_samples, n_channels))
    
    for package in range(n_packages):
        for channel in range(n_channels):
            for sample in range(n_samples):
                byte_offset = (
                    Offset
                    + (package * n_samples * n_channels + n_samples * channel + sample)
                    * 8
                )
                value = struct.unpack_from("<d", byte_data, byte_offset)[0]
                data[package * n_samples + sample, channel] = value
            data[package * n_samples : package * n_samples + n_samples , channel] = standardization(data[package * n_samples : package * n_samples + n_samples , channel])
    
    return data

def CalculateDistanceList(Temp, TargetFreqList):
    FFT_Vector = np.abs(np.fft.fft(Temp)) / len(Temp)

    Freq = np.fft.fftfreq(len(FFT_Vector), d = 1 / 512)

    FFT_Vector = FFT_Vector[: int(len(FFT_Vector) / 2)]

    Freq = Freq[: int(len(Freq) / 2)]

    Freq = Freq[np.where(Freq <= 35)]

    FFT_Vector = FFT_Vector[: len(Freq)] 

    max_freq = Freq[np.argmax(np.abs(FFT_Vector))]

    # second_freq = Freq[np.argsort(np.abs(FFT_Vector))[-2]]

    # print(max_freq, second_freq)

    DistanceList = np.abs(np.array(TargetFreqList) - max_freq) + 1e-9

    DistanceList = DistanceList / DistanceList.max()

    return DistanceList

def FFT(data):
    FFT_Vector = np.abs(np.fft.fft(data)) / len(data)

    Freq = np.fft.fftfreq(len(FFT_Vector), d = 1 / 512)

    FFT_Vector = FFT_Vector[: int(len(FFT_Vector) / 2)]

    Freq = Freq[: int(len(Freq) / 2)]

    Freq = Freq[np.where(Freq <= 35)]

    FFT_Vector = FFT_Vector[: len(Freq)] 
    return Freq, FFT_Vector

def classify(Temp, Temp_2):
    
    DoubeFreqList = [2 * i for i in FreqList]

    # DistanceList = CalculateDistanceList(Temp, FreqList)
    # DistanceList = (DistanceList * WeightList)

    Freq, FFT_Vector = FFT(Temp)
    HeatList = np.array([])
    for i in FreqList:
        Engery = 0
        for j in range(3):
            Index = int(np.argmin(np.abs(Freq - i) + j -1))
            if(Engery < FFT_Vector[Index]):
                Engery = FFT_Vector[Index]
        HeatList = np.append(HeatList, Engery)

    # index = np.argmin(np.abs(DistanceList))
    
    # print(DistanceList)

    # HeatList = (1 / np.array(DistanceList))
    HeatList = HeatList/HeatList.max()
    
    index = np.argmax(HeatList)

    return index, HeatList

def RealTimeFigurePlot(Signal):
    plt.clf()
    # Time Domain
    # plt.subplot(221)
    
    X = (np.array(range(len(Signal)))/512)
    
    O1 = standardization(Signal[:,0])
    O2 = standardization(Signal[:,1])

    plt.plot(X, O1,label = "O1")
    plt.plot(X, O2,label = "O2")
    plt.title("Time Domain", fontweight="bold", fontsize=20)
    plt.xlabel("Time(s)", fontweight="bold")
    plt.ylabel("Amplitude", fontweight="bold")
    plt.legend()

    plt.pause(0.001)
    plt.ioff()

def FinalPlot(Signal,HeatList):
    plt.clf()

    # Time Domain
    plt.subplot(221)
    plt.title("Time Domain", fontweight="bold", fontsize=20)
    # plt.plot(np.array(range(len(data))) / 512, data)
    
    X = (np.array(range(len(Signal)))/512)
    O1 = standardization(Signal[:,0])
    O2 = standardization(Signal[:,1])

    plt.plot(X, O1 ,label = "O1",linewidth = 1)
    plt.plot(X, O2,label = "O2",linewidth = 1)
    plt.legend()

    plt.xlabel("Time(s)", fontweight="bold")
    plt.ylabel("Amplitude", fontweight="bold")

    data = np.mean(Signal, axis = 1)

    # Freq Domain
    plt.subplot(222)

    FFT_Vector = np.abs(np.fft.fft((data))) / len(data)
    Freq = np.fft.fftfreq(len(FFT_Vector), d=1 / 512)
    FFT_Vector = FFT_Vector[: int(len(FFT_Vector) / 2)]
    Freq = Freq[: int(len(Freq) / 2)]
    Freq = Freq[np.where(Freq <= 35)]
    FFT_Vector = FFT_Vector[: len(Freq)]
    plt.title("Frequency Domain", fontweight="bold", fontsize=20)
    plt.plot(Freq, FFT_Vector)
    plt.xlabel("Frequency(Hz)", fontweight="bold")
    plt.ylabel("Amplitude", fontweight="bold")

    # HeatMap
    plt.subplot(223)
    plt.title("Prediction Result", fontweight="bold", fontsize=20)

    plt.bar(range(len(HeatList)),HeatList,color=['aquamarine', 'dodgerblue','royalblue','turquoise'])
    plt.xlabel("Class", fontweight="bold")
    plt.ylabel("Prediction Coefficient", fontweight="bold")
    plt.grid(False) 
    plt.xticks(range(4),[str(i)+"Hz" for i in FreqList])

    plt.subplot(224)
    plt.specgram(np.mean(Signal, axis = 1),Fs=512,cmap = "winter")
    plt.title("Specgram", fontweight="bold", fontsize=20)
    plt.xlabel("Time(s)", fontweight="bold")
    plt.ylabel("Frequency(Hz)", fontweight="bold")

    #     # Freq Domain
    # plt.subplot(223)
    
    # Temp, Temp_2 = Filters(data)
    
    # data = Temp
    
    # FFT_Vector = np.abs(np.fft.fft(data)) / len(data)
    # Freq = np.fft.fftfreq(len(FFT_Vector), d=1 / 512)
    # FFT_Vector = FFT_Vector[: int(len(FFT_Vector) / 2)]
    # Freq = Freq[: int(len(Freq) / 2)]
    # Freq = Freq[np.where(Freq <= 35)]
    # FFT_Vector = FFT_Vector[: len(Freq)]
    # plt.title("Frequency Domain (Filtered)", fontweight="bold", fontsize=20)
    # plt.plot(Freq, FFT_Vector)
    # plt.xlabel("Frequency(Hz)", fontweight="bold")
    # plt.ylabel("Amplitude", fontweight="bold")

    # plt.subplot(236)
    # plt.specgram(data,Fs=512,cmap = "winter")
    # plt.title("Specgram (Filtered)", fontweight="bold", fontsize=20)
    # plt.xlabel("Time", fontweight="bold")
    # plt.ylabel("Frequency(Hz)", fontweight="bold")
    plt.pause(0.1)
    plt.ioff()

def Filters(data):
    # data = scipy.signal.medfilt(data)
    Temp = np.zeros(len(data))
    Temp_2 = np.zeros(len(data))

    for j in FreqList:
        ############### Base Freq ####################
        b, a = scipy.signal.butter(
            4,
            [
                ((j - 1) / (512 / 2)),
                ((j + 1) / (512 / 2)),
            ],
            "bandpass",
        )
        Temp += (scipy.signal.filtfilt(b, a, standardization(data)))

        ############### Double Freq ####################
        b, a = scipy.signal.butter(
            4,
            [
                (((2 * j) - 0.5) / (512 / 2)),
                (((2 * j) + 0.5) / (512 / 2)),
            ],
            "bandpass",
        )
        Temp_2 +=  standardization(scipy.signal.filtfilt(b, a, standardization(data)))

    Temp = scipy.signal.medfilt(Temp)
    Temp = standardization(Temp)
    Temp_2 = scipy.signal.medfilt(Temp_2)
    Temp_2 = standardization(Temp_2)

    return Temp, Temp_2

def Recognition():
    global AllDataBytes, PackageNum, Headers, Predict, Offset
    AllDataBytes = bytes()
    PackageNum = 0
    Predict = -1

    ################# SampleData ##############
    SignalMatrix = np.zeros(1)
    
    TempTime = time.time()
    TempByte = bytes()
    while (len(SignalMatrix) < SampleTime * 512):
        
        print("Sampling: ", int(100 * (len(SignalMatrix)) / (SampleTime * 512)), "%","Delta Time: ",1000*(time.time() - TempTime),"ms")
        NewByte = read_data()
        # if(NewByte[-1:] == b"#"):
        #     print("End of a package")
        if(TempByte != NewByte):
            AllDataBytes += NewByte
        SignalMatrix = decode_data(AllDataBytes, 2, 32, PackageNum)
        if len(SignalMatrix) > 10:
            RealTimeFigurePlot(SignalMatrix)
        PackageNum += 1
        TempByte = NewByte
        # TempTime = time.time()

    AllDataLength = len(AllDataBytes)

    DataPackageNum = PackageNum - 1

    print(len(AllDataBytes), DataPackageNum, int((AllDataLength - 32) / 8))
    if(Headers == {}):
        Headers = decode_header(AllDataBytes)
    
    print(Headers)
    
    SignalMatrix = decode_data(
        AllDataBytes,
        int(Headers["num_channels"]),
        int(Headers["samples_per_chunk"]),
        DataPackageNum,
    )
    # print(SignalMatrix)
    print(SignalMatrix.shape)

    # SignalDisplay(SignalMatrix[-20:,0])

    data = np.mean(SignalMatrix, axis=1)

    ################# Filtering ######################
    # Temp, Temp_2 = Filters(data)
    # Predict, HeatList = CCA.cca_classify(FreqList, (np.array(data)), template = False)
    
    Predict, HeatList = FBCCA.fbcca_classify(FreqList, (np.array(data)), template=False)

    return Predict, SignalMatrix , HeatList

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def standardization(data):
    mu = np.mean(data)
    sigma = np.std(data)
    return (data - mu) / (sigma + 1e-10)

def SendResults(Predict):
    # TODO: Send Result Through COM Port
    LetterList = ["D","A","B","C"]
    for i in range(4):
        send_data_to_serial(i, LetterList[Predict])

def send_data_to_serial(index, data):
    # 设置串口信息
    port_list = ['COM5', 'COM7', 'COM8', 'COM16']
    baud_rate = 115200
    # 检查传入的索引是否有效
    if index < 0 or index >= len(port_list):
        print("Invalid index!")
        return
    try:
        # 打开指定索引对应的串口
        ser = serial.Serial(port_list[index], baud_rate)
        print("Sending data",data,"to serial port", port_list[index])
        ser.write(data.encode())
        ser.close()
    except Exception as e:
        print("Serial port error:", str(e))

if __name__ == "__main__":
    # for i in range(4):
        # send_data_to_serial(i,"D")
    # SendResults(1)
    # sys.exit(0)

    global signal_socket
    fig, ax = plt.subplots(figsize=(14, 10), dpi = 50)
    fig.set_tight_layout(True)
    plt.ion()
    Offset = 32
    RoundCount = 0
    for i in SendingList:
        ConnectOpenViBE()
        Headers = {}
        if(Headers == {}):
            Offset = 32
        else:
            Offset = 0
        Result, SignalMatrix, HeatList = Recognition()
        
        print("Prediction: ", "Index:", Result, " Freq:", FreqList[Result], "Hz")
        
        FinalPlot(SignalMatrix, HeatList)
        
    
        plt.savefig(
            "../Log/RealTest/"
            + str(SampleTime)
            + "s"
            +" - "
            + str(FreqList[Result]) + "Hz"
            + " - "
            + str(time.strftime("%Y-%m-%d",time.gmtime()))
            + " - "
            + str(int(time.time()))
            + ".png",
            dpi = 400,
        )

        signal_socket.close()
        ############## Sending Data #################
        SendResults(SendingList[RoundCount])
        
        RoundCount += 1
        
        print("Break")
        
        time.sleep(BreakTime)
    plt.show()
