clc;
clear;
% 读取浊音语音帧
[x, Fs] = audioread('cut_audio.wav');

% 2. 分帧
frameSize = round(0.02 * Fs); % 20ms帧长
overlap = round(0.01 * Fs);    % 10ms帧移
frames = enframe(x, frameSize, overlap);
reconstructed_frames=zeros(size(frames));
% 3. 计算LPC系数
p = 10; % LPC阶数
A = zeros(p, size(frames, 2));
bits=4;
for i = 1:size(frames, 1)
    frame=frames(i,:)';
    % 应用窗函数（这里使用汉宁窗）
    windowed_frame = frame .* hann(length(frame));
    % 计算 LPC 系数
    [a, e] = lpc(windowed_frame, p);
    % 计算预测残差信号
    residual = filter(a, 1, frame);
    max_residual = max(abs(residual));
    max_a = max(abs(a));
    residual1=residual/max_residual;
    a1=a/max_a;
    [encoded_residual,encoded_A]=lpc_encoder(a1,residual1,bits);
    [decoded_residual,decoded_A]=lpc_decoder(encoded_residual,encoded_A,bits);
    decoded_residual1=decoded_residual*max_residual;
    decoded_A1=decoded_A*max_a;
    % 通过LPC参数滤波重构语音信号
    reconstructed_frame=filter(1, decoded_A1, decoded_residual1)';
    reconstructed_frames(i,:) = frame;
end
reconstructed_signal=ola(reconstructed_frames, frameSize,overlap);
reconstructed_signal=reconstructed_signal';
% 计算信噪比
snr_values = calculateSNR(x, reconstructed_signal(1:size(x,1),:));
% 输出信噪比
disp('LPC SNR values(4 bit):');
disp(snr_values);

% 比较重构信号和原信号
figure;
subplot(2,1,1);
plot(x); % 原始语音信号
title('Original voiced frame');
xlabel('Sample');
ylabel('Amplitude');
% 
% % 使用逻辑索引将大于1的值设置为1
% reconstructed_signal(reconstructed_signal > 128) = 128;
% 
% % 使用逻辑索引将小于-1的值设置为-1
% reconstructed_signal(reconstructed_signal < -128) = -128;

% 指定文件名
filename = ['lpc(p=' num2str(p) ',bit=' num2str(bits) ').wav'] ;
 
% 保存WAV文件
audiowrite(filename, reconstructed_signal(1:size(x,1),:), Fs);  

subplot(2,1,2);
plot(reconstructed_signal(1:size(x,1),:)); % 重构的语音信号
title('Reconstructed voiced frame');
xlabel('Sample');
ylabel('Amplitude');


function output_value = double_to_int_bits(input_value, bits)
    % 输入参数 input_value 是一个64位有符号小数
    % bits 是所需的比特数，可以是4、6或10
    % 返回值 output_value 是一个有符号整数
    
    % 将输入值缩放到目标比特数的范围内
    range = 2^(bits-1) - 1; % 范围为[-range, range]
    scaled_value = input_value * range;
    
    % 四舍五入到最接近的整数
    rounded_value = round(scaled_value);
    
    % 确保值在目标范围内
    if rounded_value > range
        output_value = range;
    elseif rounded_value < -range
        output_value = -range;
    else
        output_value = int8(rounded_value);
    end
end
function output_value = int_to_double_bits(input_value, bits)
    % 输入参数 input_value 是一个有符号整数
    % bits 是所用的比特数，可以是4、6或10
    % 返回值 output_value 是一个64位有符号小数
    
    % 将输入值缩放到目标范围内
    range = 2^(bits-1) - 1; % 范围为[-range, range]
    scaled_value = double(input_value) / range;
    
    % 返回缩放后的值
    output_value = scaled_value;
end
function [encoded_residual,encoded_A]=lpc_encoder(A,residual,bits)
    %残差编码
    encoded_residual = double_to_int_bits(residual, bits);
    %a编码
    encoded_A = double_to_int_bits(A, bits);   
end
function [decoded_residual,decoded_A]=lpc_decoder(encoded_residual,encoded_A,bits)
    %残差编码
    decoded_residual = int_to_double_bits(encoded_residual, bits);
    %a编码
    decoded_A = int_to_double_bits(encoded_A, bits);   
end

function x_reconstruct = ola(frames,win,inc)
    % frames: 分帧后的数据矩阵，每一列是一个帧
    % win: 窗函数，可以是向量或标量（如果未指定窗函数）
    % inc: 帧移

    nwin = length(win); % 窗长
    nf = size(frames, 1); % 帧数
    len = nwin; % 帧长，这里假设帧长等于窗长，如果不是，需要进行调整

    % 初始化重构的信号，长度至少与原始信号相同
    x_reconstruct = zeros(1, win + len * inc * nf);

    % 计算每个帧的起始和结束位置
    ind_start = (0:nf-1) * inc+1;
    ind_end = ind_start + win - 1;

    % 对每个帧进行OLA处理
    for k = 1:nf
        % 应用窗函数
        % frame_k = frames(k, :) .* win;
        frame_k = frames(k, :) ;
        
        % 将加窗后的帧添加到重构信号中
        x_reconstruct(ind_start(k):ind_end(k)) =  frame_k;
    end
end
function snr = calculateSNR(originalSignal, noisySignal)
    % 计算原始信号的功率
    originalPower = sum(abs(originalSignal).^2) / length(originalSignal);

    % 计算噪声信号的功率
    noisyPower = sum(abs(noisySignal - originalSignal).^2) / length(noisySignal);

    % 计算信噪比（dB）
    snr = 10 * log10(originalPower / noisyPower);
end

