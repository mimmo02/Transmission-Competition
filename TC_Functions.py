import numpy as np
import matplotlib.pyplot as plt
import argparse

# computation of the entropy of a given text
def get_info_text(text):

    # Perform the Entropy and Entropy of a uniform distribution of the text
    unique, counts = np.unique(list(text), return_counts=True)
    probabilities = counts/np.sum(counts)
    entropy = np.sum(probabilities * np.log2(1/probabilities))
    entropy_uniform = np.log2(len(unique))

    # Perform the Redundancy of the text
    redundancy = 1-(entropy/entropy_uniform)

    # Print the Entropy, Entropy of a uniform distribution and Redundancy
    print('@source_coding_layer->Entropy:', entropy)
    print('@source_coding_layer->Entropy of a uniform distribution:', entropy_uniform)
    print('@source_coding_layer->Redundancy:', redundancy)

    return 

# SOURCE CODING LAYER

# Function that encode a source code with the Huffman algorithm
def huffman_encoding(text):
    print('@source_coding_layer.HFM-> HUFFMAN ENCODING ////////////////////////////////////////////////')
    # Perform the Entropy and Entropy of a uniform distribution of the text
    unique, counts = np.unique(list(text), return_counts=True)
    probabilities = counts/np.sum(counts)

    # Create a dictionary with the characters and their probabilities
    dictionary = {char: prob for char, prob in zip(unique, probabilities)}
          
    # Create a list with the nodes of the Huffman tree
    nodes = []
    for key in dictionary.keys():
        nodes.append([key, dictionary[key]])

    # Create a dictionary to store the codes
    codes = {char: '' for char in dictionary.keys()}
    # Create the Huffman tree starting from lowest probabilities and find encoded codes for each character
    while len(nodes) > 1:
        # Sort the nodes by the probabilities
        nodes = sorted(nodes, key=lambda x: x[1])
        # Get the two nodes with the lowest probabilities
        higher = nodes[0]
        lower = nodes[1]
        # Remove the nodes from the list
        nodes = nodes[2:]
        # Create a new node with the sum of the probabilities of the two nodes
        for char in higher[0]:
            # Add '0' to the code of the characters of the left node
            codes[char] = '0' + codes[char] # Added before because the code is reversed
        for char in lower[0]:
            # Add '1' to the code of the characters of the right node
            codes[char] = '1' + codes[char] # Added before because the code is reversed
        nodes.append([higher[0] + lower[0], higher[1] + lower[1]])
    # Encode the text
    encoded_text = ''
    for char in text:
        encoded_text += codes[char]
    # Print the encoded text
    print('@source_coding_layer.HFM-> Encoded text:', encoded_text)
    print('@source_coding_layer.HFM-> Huffman dictionary:', codes)
    return encoded_text, codes

# Function that receivs a data and decode with the Huffman algorithm
def huffman_decoding(encoded_text, dictionary):
    # Create a dictionary to store the characters and their codes
    dictionary = {dictionary[char]: char for char in dictionary.keys()}
    print(dictionary)
    print('@source_decoding_layer.HFM-> HUFFMAN DECODING ////////////////////////////////////////////////')
    # Decode the text
    decoded_text = ''
    code = ''
    for bit in encoded_text:
        code += bit
        if code in dictionary.keys():
            decoded_text += dictionary[code]
            code = ''
    return decoded_text

# Encoding LZW algorithm
def LZW_encoding(text):
    print('@source_coding_layer.LZW-> LZW ENCODING ////////////////////////////////////////////////')
    # Initialization of the dictionary based on the text in order of appearance {character: index}
    dictionary = {text[0]: 0}
    for i in range(1, len(text)):
        if text[i] not in dictionary:
            dictionary[text[i]] = len(dictionary)
    # Encode the text and expand the dictionary till 12 bits (4096 words)
    w = ""
    encoded_text = []
    for c in text: # For each character in the text
        wc = w+c  # Add the character to the current word
        if wc in dictionary: # If the word is in the dictionary
            w = wc # Update the current word
        else:
            # if 4096 words are in the dictionary, stop adding new words in the dictionary but continue to encode the text
            if len(dictionary) == 4096:
                encoded_text.append(dictionary[w])
                w = c
            else:
                encoded_text.append(dictionary[w]) # if the new word is not in the dictionary, add the code of the passed word to the encoded text
                dictionary[wc] = len(dictionary) # Add the new word to the dictionary
                w = c # Update the current word with the single character (shift of one position)
    if w: # If there is a word left
        encoded_text.append(dictionary[w]) # Add the code of the current word to the dictionary
    # Print the encoded text
    print('@source_coding_layer.LZW-> Encoded text:', encoded_text)
    print('@source_coding_layer.LZW-> Dictionary:', dictionary)
    return encoded_text, dictionary

# Decode the LZW code
def LZW_decoding(code, dictionary):
    print('@source_decoding_layer.LZW-> LZW DECODING ////////////////////////////////////////////////')
    # Create a dictionary to store the characters and their codes and invert keys and values
    dictionary = {dictionary[key]: key for key in dictionary.keys()}
    # Decode the text
    text = ""
    for c in code:
        text += dictionary[c]
    return text

# V2 of the source layer functions

# Encapsulation of the source coding layer
def encapsulation_source_coding_layer(method, encoded_text, dictionary): 
    if method == 0:
        # Convert dictionary in a binary string with separators: ¢ for key-value and § for new line
        dict_str = ""
        for key, value in dictionary.items():
            # se è l'ultimo elemento non mettere il carattere di new line
            if key == list(dictionary.keys())[-1]:
                dict_str += key + "¢" + str(value)
            else:
                dict_str += key + "¢" + str(value) + "§"
        # String to binary 1-bit packet of the dictionary+encoded text serialized (one array)
        encoded_text = [int(i) for i in encoded_text]
        dictionary = ''.join(format(ord(i), '08b') for i in dict_str)
        dictionary = [int(i) for i in dictionary]
        # Put all together
        converted_data = dictionary + encoded_text
        # Add Header to the data package (Header: 32bits = Huffman/LZW 1bit + length dictionary 17bits + length text 14bits) (list to string)
        header = [0] + [int(bit) for bit in ''.join(format(len(dictionary), '017b'))] + [int(bit) for bit in ''.join(format(len(encoded_text), '014b'))]
        encapsulated_data = header + converted_data

    elif method == 1:
        # Convert dictionary in a binary string with separators: ¢ for key-value and § for new line
        dict_str = ""
        for key, value in dictionary.items():
            # se è l'ultimo elemento non mettere il carattere di new line
            if key == list(dictionary.keys())[-1]:
                dict_str += key + "¢" + str(value)
            else:
                dict_str += key + "¢" + str(value) + "§"

        # String to binary 1-bit packet of the dictionary+encoded text serialized (one array)
        encoded_text = ''.join([format(i, '012b') for i in encoded_text])
        encoded_text = [int(i) for i in encoded_text]
        dictionary = ''.join(format(ord(i), '012b') for i in dict_str)
        dictionary = [int(i) for i in dictionary]

        # Put all together
        converted_data = dictionary + encoded_text
        # Add Header to the data package (Header: 32bits = Huffman/LZW 1bit + length dictionary 17bits + length text 14bits)
        header = [1] + [int(bit) for bit in ''.join(format(len(dictionary), '017b'))] + [int(bit) for bit in ''.join(format(len(encoded_text), '014b'))]
        encapsulated_data = header + converted_data
    return encapsulated_data

# Decapsulation of the source coding layer
def decapsulation_source_coding_layer(encapsulated_data):
    # Extract the header
    header = encapsulated_data[:32]
    # Get the method
    method = header[0]
    # Get the length of the dictionary
    length_dict = int(''.join(map(str, header[1:18])), 2)
    # Get the length of the text
    length_text = int(''.join(map(str, header[18:])), 2)
    # Get the dictionary
    dictionary = encapsulated_data[32:32+length_dict]
    # Get the text
    encoded_text = encapsulated_data[32+length_dict:32+length_dict+length_text]

    if method == 0:
        # cast to string type and binary to char
        encoded_text = ''.join([str(i) for i in encoded_text])
        dictionary = ''.join([str(i) for i in dictionary])
        dictionary = ''.join(chr(int(dictionary[i:i+8], 2)) for i in range(0, len(dictionary), 8))
        print(encoded_text)
        print(dictionary)
        # reconstruct the dictionary
        final_dictionary = {}
        for line in dictionary.split("§"):
            if line:
                key, value = line.split("¢")
                final_dictionary[key] = value
        return encoded_text, final_dictionary, method
    else:
        # convert encoded_text to array type with integer (every 12bits)
        encoded_text = ''.join([str(i) for i in encoded_text])
        encoded_text = [int(encoded_text[i:i+12], 2) for i in range(0, len(encoded_text), 12)]
        # cast to string type and binary to char
        dictionary = ''.join([str(i) for i in dictionary])
        dictionary = ''.join(chr(int(dictionary[i:i+12], 2)) for i in range(0, len(dictionary), 12))
        # reconstruct the dictionary
        final_dictionary = {}
        for line in dictionary.split("§"):
            if line:
                key, value = line.split("¢")
                final_dictionary[key] = int(value)
        return encoded_text, final_dictionary, method
    return

# Source encoding layer function
def source_encoding(text):
    # Import text from file and print it
    print('@source_coding_layer-> Text:', text)
    # Perform huffman and LZW encoding to compare and choose best method
    encoded_text_huffman, dict_huffman = huffman_encoding(text)
    encoded_text_LZW, dict_LZW = LZW_encoding(text)
    # Compression ratio
    compression_ratio(text, dict_huffman, dict_LZW)
    # Encapsulate data both methods
    encapsulated_data_source_H = encapsulation_source_coding_layer(0, encoded_text_huffman, dict_huffman)
    encapsulated_data_source_L = encapsulation_source_coding_layer(1, encoded_text_LZW, dict_LZW)
    # Get length of the two methods
    Huffman_length = len(encapsulated_data_source_H)
    print('@source_coding_layer-> Huffman length:', Huffman_length, 'bits')
    LZW_length = len(encapsulated_data_source_L)
    print('@source_coding_layer-> LZW length:', LZW_length, 'bits') 
    # Choose the best method
    if Huffman_length < LZW_length:
        print('@source_coding_layer-> Huffman encoding is the best method')
        encapsulated_data_source = encapsulated_data_source_H
    else:
        print('@source_coding_layer-> LZW encoding is the best method')
        encapsulated_data_source = encapsulated_data_source_L
    # Encapsulate data
    print('@source_coding_layer-> Source Encoding Data:', encapsulated_data_source)
    return encapsulated_data_source


# Source decoding layer function using convert_C_to_S_Huffman and convert_C_to_S_LZW
def source_decoding(data_received):
    encoded_text, dictionary, method = decapsulation_source_coding_layer(data_received)
    if method == 0:
        # Decode the Huffman data
        source_decoding_data = huffman_decoding(encoded_text, dictionary)
    else:
        # Decode the LZW data
        source_decoding_data = LZW_decoding(encoded_text, dictionary)
    print('@source_decoding_layer-> Source Decoding Data:', source_decoding_data)
    return source_decoding_data

# Function that returns compression ratio
def compression_ratio(text, dict_huffman, dict_LZW):
    # Get size of the file.txt file with fixed length coding
    # Get how many unique characters are in the text
    unique, counts = np.unique(list(text), return_counts=True)
    probabilities = counts/np.sum(counts)
    # Calculate the size of a symbol in bits
    fixed_length_size = np.ceil(np.log2(len(unique)))
    # Calculate bit/symbol for LZW and Huffman
    # Huffman (sum of (probabilities of each character * its code length))
    huffman_size = np.sum([len(dict_huffman[key])*probabilities[i] for i, key in enumerate(unique)])
    # LZW (close to H0 so log2(number of indexes of the dictionary))
    LZW_size = np.log2(len(dict_LZW))
    # Calculate the compression ratio
    compression_ratio_H = huffman_size / fixed_length_size
    compression_ratio_L = LZW_size / fixed_length_size
    print('@source_coding_layer-> Compression ratio Huffman:', compression_ratio_H)
    print('@source_coding_layer-> Compression ratio LZW:', compression_ratio_L)
    return compression_ratio
# CHANNEL CODING LAYER

# notation G: generation matrix, H: parity-check matrix, D: data extraction matrix
# Matrix_CodewordSize_DataSize

# size = 11x15
G_15_11 = [[1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
           [1,0,0,1,1,0,0,0,0,0,0,0,0,0,0],
           [0,1,0,1,0,1,0,0,0,0,0,0,0,0,0],
           [1,1,0,1,0,0,1,0,0,0,0,0,0,0,0],
           [1,0,0,0,0,0,0,1,1,0,0,0,0,0,0],
           [0,1,0,0,0,0,0,1,0,1,0,0,0,0,0],
           [1,1,0,0,0,0,0,1,0,0,1,0,0,0,0],
           [0,0,0,1,0,0,0,1,0,0,0,1,0,0,0],
           [1,0,0,1,0,0,0,1,0,0,0,0,1,0,0],
           [0,1,0,1,0,0,0,1,0,0,0,0,0,1,0],
           [1,1,0,1,0,0,0,1,0,0,0,0,0,0,1]]

# size = 15x4
H_15_11 = [[1,0,0,0],
           [0,1,0,0],
           [1,1,0,0],
           [0,0,1,0],
           [1,0,1,0],
           [0,1,1,0],
           [1,1,1,0],
           [0,0,0,1],
           [1,0,0,1],
           [0,1,0,1],
           [1,1,0,1],
           [0,0,1,1],
           [1,0,1,1],
           [0,1,1,1],
           [1,1,1,1]]

# size = 15x11
D_15_11 = [[0,0,0,0,0,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0,0,0],
           [1,0,0,0,0,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0,0,0],
           [0,1,0,0,0,0,0,0,0,0,0],
           [0,0,1,0,0,0,0,0,0,0,0],
           [0,0,0,1,0,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0,0,0],
           [0,0,0,0,1,0,0,0,0,0,0],
           [0,0,0,0,0,1,0,0,0,0,0],
           [0,0,0,0,0,0,1,0,0,0,0],
           [0,0,0,0,0,0,0,1,0,0,0],
           [0,0,0,0,0,0,0,0,1,0,0],
           [0,0,0,0,0,0,0,0,0,1,0],
           [0,0,0,0,0,0,0,0,0,0,1]]

# size = 4x7
G_7_4 = [[1,1,1,0,0,0,0],
         [1,0,0,1,1,0,0],
         [0,1,0,1,0,1,0],
         [1,1,0,1,0,0,1]]

# size = 7x3
H_7_4 = [[1,0,0],
         [0,1,0],
         [1,1,0],
         [0,0,1],
         [1,0,1],
         [0,1,1],
         [1,1,1]]

# size = 7x4
D_7_4 = [[0,0,0,0],
         [0,0,0,0],
         [1,0,0,0],
         [0,0,0,0],
         [0,1,0,0],
         [0,0,1,0],
         [0,0,0,1]]

def channel_code_gen(data, G):
    # data: binary data
    # G: generation matrix
    # return: codeword
    data = np.array(data)
    codeword = np.dot(data, G) % 2
    return codeword

def channel_decode_gen(codeword, H, D):
    # code: binary codeword
    # H: parity-check matrix
    # D: data extraction matrix
    # return: data
    m = len(H[0][:]) # number of parity bits
    c= np.array(codeword)
    p = np.dot(c, H) % 2
    error = False
    if p.any():
        error_position = np.sum([2**i for i in range(m) if p[i] == 1])
        # flip the bit in the error position
        c[error_position-1] = 1 - c[error_position-1]
        error = True
               
    data = np.dot(c, D) % 2
    return data,error

def split_packet(message, packet_size):
    # message: binary data
    # packet_size: size of each packet
    # return: list of packets
    print("@channel_coding_layer-> SPLITTING DATA ////////////////////////////////////////////////")
    packets = []
    for i in range(0, len(message), packet_size):
        packets.append(message[i:i+packet_size])
    # fill the last packet with zeros
    if len(packets[-1]) < packet_size:
        fill_bits = packet_size - len(packets[-1])
        fill = [0]*fill_bits
        packets[-1] = packets[-1] + fill
        # cast fill_bits to binary
        fill_bits = [int(x) for x in list(bin(fill_bits)[2:])]
        # dimension of the fill_bits equal to packet_size
        fill_bits = [0]*(packet_size-len(fill_bits)) + fill_bits
        # push in first position an addictional packet with the number of zeros added
        packets.insert(0, fill_bits)
    else:
        packets.insert(0, [0]*packet_size)
    print("@channel_coding_layer-> Packets: ", packets)
    return packets

def channel_coding(MSG,Hamming_size): 
    print("@channel_coding_layer-> ENCODING DATA ////////////////////////////////////////////////")
    if Hamming_size == "15x11":
        packet_size = 11
        G = G_15_11
    elif Hamming_size == "7x4":
        packet_size = 4
        G = G_7_4
    packets = split_packet(MSG, packet_size)
    codewords = []
    print("@channel_coding_layer-> ENCODING DATA PACKETS ////////////////////////////////////////")
    for packet in packets:
        codeword = channel_code_gen(packet, G)
        codewords.append(codeword)
        
    return codewords

def channel_decoding(COD,Hamming_size):
    print("@channel_coding_layer-> DECODING DATA PACKETS ////////////////////////////////////////")
    if Hamming_size == "15x11":
        H = H_15_11
        D = D_15_11
    elif Hamming_size == "7x4":
        H = H_7_4
        D = D_7_4
    packets = []
    errors = 0
    for codeword in COD:
        packet,error = channel_decode_gen(codeword, H, D)
        if error:
            errors += 1
        packets.append(packet)
    print("@channel_coding_layer-> Decoded Packets: ", packets)
    # first packet is the number of zeros added
    fill_bits = packets.pop(0)
    # binary to decimal
    fill_bits = int(''.join([str(x) for x in fill_bits]), 2)
    # unite the packets
    packets = [item for sublist in packets for item in sublist]
    return packets,fill_bits,errors

# Conversion functions between source coding and channel coding

# Prepare data to go from source coding layer to channel coding layer
def convert_S_to_C_Huffman(SET_TX, SED_TX):
    # cast to binary array type
    # Convert dictionary in a binary string
    # notation: binary ascii character + "\t" + binary code + "\n"
    dict_str = ""
    for key, value in SED_TX.items():
        # se è l'ultimo elemento non mettere il carattere di new line
        if key == list(SED_TX.keys())[-1]:
            dict_str += key + "¢" + str(value)
        else:
            dict_str += key + "¢" + str(value) + "§"
    
    # String to binary
    SET_TX = [int(i) for i in SET_TX]
    SED_TX = ''.join(format(ord(i), '08b') for i in dict_str)
    SED_TX = [int(i) for i in SED_TX]
    return SET_TX, SED_TX

# Prepare data to go from the channel coding layer to the source coding layer
def convert_C_to_S_Huffman(SET_RX, SED_RX, fill_bit_number_T, fill_bit_number_D):
    # Remove fill bits from arrays
    SET_RX = SET_RX[:(len(SET_RX)-fill_bit_number_T)]
    SED_RX = SED_RX[:(len(SED_RX)-fill_bit_number_D)]

    # cast to string type and binary to char
    SET_RX = ''.join([str(i) for i in SET_RX])
    SED_RX = ''.join([str(i) for i in SED_RX])
    SED_RX = ''.join(chr(int(SED_RX[i:i+8], 2)) for i in range(0, len(SED_RX), 8))

    # reconstruct the dictionary
    dictionary = {}
    for line in SED_RX.split("§"):
        if line:
            key, value = line.split("¢")
            dictionary[key] = value
    return SET_RX, dictionary

# Prepare data to go from source coding layer to channel coding layer
def convert_S_to_C_LZW(SET_TX, SED_TX):
    # cast to binary array type
    # Convert dictionary in a binary string
    # notation: binary ascii character + "\t" + binary code + "\n"
    dict_str = ""
    for key, value in SED_TX.items():
        # se è l'ultimo elemento non mettere il carattere di new line
        if key == list(SED_TX.keys())[-1]:
            dict_str += key + "¢" + str(value)
        else:
            dict_str += key + "¢" + str(value) + "§"
    print(dict_str)
    # There are empty lines in the dict_str, remove them im not sure that is double \n
    
    print(dict_str)
    # Array to binary use 12bits for each element of the array SET_TX
    SET_TX = ''.join([format(i, '012b') for i in SET_TX])
    SET_TX = [int(i) for i in SET_TX]

    # String to binary for SED_TX
    SED_TX = ''.join(format(ord(i), '012b') for i in dict_str)
    SED_TX = [int(i) for i in SED_TX]
    return SET_TX, SED_TX

# Prepare data to go from the channel coding layer to the source coding layer
def convert_C_to_S_LZW(SET_RX, SED_RX, fill_bit_number_T, fill_bit_number_D):
    # Remove fill bits from arrays
    SET_RX = SET_RX[:(len(SET_RX)-fill_bit_number_T)]
    SED_RX = SED_RX[:(len(SED_RX)-fill_bit_number_D)]
    
    # convert SET_RX to array type with integer (every 8bits)
    SET_RX = ''.join([str(i) for i in SET_RX])
    SET_RX = [int(SET_RX[i:i+12], 2) for i in range(0, len(SET_RX), 12)]
    # cast to string type and binary to char
    SED_RX = ''.join([str(i) for i in SED_RX])
    SED_RX = ''.join(chr(int(SED_RX[i:i+12], 2)) for i in range(0, len(SED_RX), 12))
    print(SED_RX)
    # reconstruct the dictionary
    dictionary = {}
    for line in SED_RX.split("§"):
        if line:
            key, value = line.split("¢")
            dictionary[key] = int(value)
    return SET_RX, dictionary

# encapsulate the Channel Coding Layer data
def channel_coding_encapsulate(CCL_data, Hamming_size):
    # serialize the data
    np.array(CCL_data).ravel()
    # add the Header: 0 = H(15x11), 1 = H(7x4)
    if Hamming_size == "15x11":
        header = [0,0,0,0,0,0,0,0,0,0]
    elif Hamming_size == "7x4":
        header = [1,1,1,1,1,1,1,1,1,1]
    # add the header to the data
    CCL_block = np.insert(CCL_data, 0, header)
    return CCL_block
 
# decapsulate the Channel Coding Layer data
def channel_coding_decapsulate(CCL_block):
    # remove the header
    header = CCL_block[0:9]
    CCL_block = CCL_block[10:]
    header_mean = np.mean(header)
    if header_mean < 0.5:
        Hamming_size = "15x11"
        # check if the data is divisible by 15
        if len(CCL_block) % 15 != 0:
            # add zeros to the end of the data
            zeros = 15 - len(CCL_block) % 15
            CCL_block = np.append(CCL_block, np.zeros(zeros))
        CCL_data = np.array(CCL_block).reshape(-1, 15)
        CCL_data = CCL_data.astype(int)
    elif header_mean >= 0.5:
        Hamming_size = "7x4"
        # check if the data is divisible by 7
        if len(CCL_block) % 7 != 0:
            # add zeros to the end of the data
            zeros = 7 - len(CCL_block) % 7
            CCL_block = np.append(CCL_block, np.zeros(zeros))
        CCL_data = np.array(CCL_block).reshape(-1, 7)
        CCL_data = CCL_data.astype(int)
   
    # return the data and the header
    return CCL_data, Hamming_size


#AM
import numpy as np
from scipy.signal import butter, filtfilt

def PLL_function(input_signal, fc, phase_gain, freq_gain, fs):
    # Calculate the normalized angular frequency
    angular_frequency = 2 * np.pi * fc / fs
    current_phase = 0
    frequency_offset = 0
    
    # Initialize output and control arrays
    output_signal = np.zeros(len(input_signal))
    control_signal = np.zeros(len(input_signal))

    for i in range(len(input_signal)):
        # Generate the output signal based on the current phase
        output_signal[i] = np.cos(current_phase)
        
        # Calculate the phase offset based on the input signal
        phase_offset = -input_signal[i] * np.sin(current_phase)
        
        # Update the frequency offset
        frequency_offset += freq_gain * phase_offset
        
        # Calculate the control signal
        control_signal[i] = phase_gain * phase_offset + frequency_offset
        
        # Update the current phase
        current_phase += angular_frequency + control_signal[i]
        
        # Normalize the phase to the range [-pi, pi]
        current_phase = (current_phase + np.pi) % (2 * np.pi) - np.pi

    return output_signal

def am_modulate(binary_message, fs = 44e3, fc = 10e3,  num_cycles = 10):

    fs = int(fs)
    fc = int(fc)

    # Convert binary message to square wave
    bit_duration = num_cycles / fc
    samples_per_bit = int(fs * bit_duration)
    
    # Create the square wave with the correct length
    square_wave = np.repeat(binary_message, samples_per_bit)
    
    # Generate the carrier wave with the correct length
    t = np.arange(len(square_wave)) / fs
    carrier = np.cos(2 * np.pi * fc * t)
    
    # Modulate the square wave
    modulated_signal = (1 + square_wave)/2 * carrier      # bit 0 -> 0.5, bit 1 -> 1
    return modulated_signal

def am_demodulate(modulated_signal, fs = 44e3, fc = 10e3, num_cycles = 10):

    fs = int(fs)
    fc = int(fc)
    
    bit_duration = num_cycles / fc
    samples_per_bit = int(fs * bit_duration)
    

    # normalize the signal amplitude by dividing by the 100th maximum value(added by mimmo02)-------------------------------------
    max = np.sort(np.abs(modulated_signal))[::-1][100]
    modulated_signal = modulated_signal / max

    # convert to mono the double channel signal if needed (added by @biofainapap)
    if len(modulated_signal.shape) > 1:
        modulated_signal = modulated_signal[:, 0]

    plt.figure()
    plt.plot(modulated_signal)
    plt.title("Normalized Modulated Signal")
    #---------------------------------------------------------------------------------------

    modulated_signal = 2*modulated_signal
    # Generate the carrier signal
    y =  PLL_function(modulated_signal, fc, 0.01, 0.01, fs)
    
    # Demodulate the signal
    demodulated_signal = modulated_signal * y
    demodulated_signal = np.abs(demodulated_signal)
    
    plt.figure()
    plt.plot(demodulated_signal)
    plt.title("Demodulated Signal")
    plt.show()
    
    # Apply low-pass filter to the demodulated signal directly here
    normal_cutoff = fc/fs
    b, a = butter(5, normal_cutoff, btype='low', analog=False)
    filtered_demodulated_signal = filtfilt(b, a, demodulated_signal)
    
  
    
    # normalize the signal amplitude (added by mimmo02)-------------------------------------
    max = np.max(np.abs(filtered_demodulated_signal))
    filtered_demodulated_signal = filtered_demodulated_signal / max
    
    
    # calculate the mean par each bit
    means = []
    for i in range(0, len(filtered_demodulated_signal), samples_per_bit):
        means.append(np.mean(filtered_demodulated_signal[i:i+samples_per_bit]))
    
    # cancel the silence bits
    # caluculate the mean of 5 bits set
    
    part_means = []
    std = []
    
    for i in range(0, len(means), 5):
        group_mean = np.mean(means[i:i+5])
        # calculate the standard deviation of the group
        group_std = np.std(means[i:i+5])
        part_means.append(group_mean)
        std.append(group_std)
        
    std_mean = np.mean(std)
    part_means_mean = np.mean(part_means)   
        
    plt.figure()
    plt.plot(part_means)
    plt.plot(std)
    plt.plot([0, len(part_means)], [part_means_mean, part_means_mean], 'r')
    plt.plot([0, len(std)], [std_mean, std_mean], 'g')
    plt.title("Means and Standard Deviation")
    plt.show()
        
    # search for std < std_mean and part_means < part_means_mean
    del_num = 0
    for i in range(0, len(means), 5):
        if std[i//5] < std_mean and part_means[i//5] < part_means_mean:
            del_num += 5
        else:
            break
        
    filtered_demodulated_signal = filtered_demodulated_signal[del_num*samples_per_bit:]
    
            

    
    mean = np.mean(filtered_demodulated_signal)
    
    
    
    # plot the means for each bit
    plt.figure()
    plt.plot(filtered_demodulated_signal)
    # plot the means for each bit
    #for i in range(min(len(filtered_demodulated_signal) // samples_per_bit, len(means))):
    #   plt.plot([i * samples_per_bit, (i + 1) * samples_per_bit], [means[i], means[i]], 'r')
    plt.plot([0, len(filtered_demodulated_signal)], [mean, mean], 'g')
    plt.title("Filtered Demodulated Signal")
    plt.show()
    
    
    
    #function to reduce the overshoot
    def sig_rescale(x,shift):
        up = np.arctan((10.0-shift)*50)
        down = np.arctan((-10.0-shift)*50)
        ampl = np.abs(up) + np.abs(down)
        y =  (np.arctan((x-shift)*50) + np.abs(down)) / ampl
        return y
    
     
    # rescale the bits depending on the mean value
    for i in range(0, len(filtered_demodulated_signal), samples_per_bit):
        #filtered_demodulated_signal[i:i+samples_per_bit] = sig_rescale(filtered_demodulated_signal[i:i+samples_per_bit], means[i//samples_per_bit])
        filtered_demodulated_signal[i:i+samples_per_bit] = sig_rescale(filtered_demodulated_signal[i:i+samples_per_bit], mean)
 
    
        

    
    plt.figure()
    plt.plot(filtered_demodulated_signal)
    plt.title("Levelled Signal")
    plt.show()
    
    #---------------------------------------------------------------------------------------
    
    # Convert the filtered demodulated signal back to binary data
    filtered_demodulated_signal = np.round(2 * filtered_demodulated_signal) - 1
    # set negative values to 0
    filtered_demodulated_signal[filtered_demodulated_signal < 0] = 0
    
    plt.figure()
    plt.plot(filtered_demodulated_signal)
    plt.title("Digitalized Demodulated Signal")
    plt.show()
    
    num_bits = len(filtered_demodulated_signal) // samples_per_bit
    binary_data = np.zeros(num_bits)

    for i in range(num_bits):
        segment = filtered_demodulated_signal[i * samples_per_bit:(i + 1) * samples_per_bit]
        binary_data[i] = 1 if np.mean(segment) > 0.5 else 0  # Use mean to determine bit value

    return binary_data.astype(int)


# add syncro bits to the data (added by mimmo02)
def syncro_bits_addition(data):
    dummy = 15*[1,0]
    bits = [1,0,1,0,1,0,1,0,1,0,1,1,1,1,1,0,0,0,0,0]    # 10 syncro bits
    syncro_bits = np.concatenate((dummy, bits))
    return np.concatenate((syncro_bits, data))


# syncro bits detection for AM (modified by mimmo02)
def syncro_bits_detection(data):
    syncro_bits_window = [1,0,1,0,1,0,1,0,1,0,1,1,1,1,1,0,0,0,0,0] # 10 syncro bits
    syncro_bits_window_len = len(syncro_bits_window)
    # find the syncro bits sliding the window
    for i in range(len(data) - syncro_bits_window_len):
        if np.all(data[i:i + syncro_bits_window_len] == syncro_bits_window):
            return data[i + syncro_bits_window_len:]
    return data
    
# Chirp
def chirp_modulate(data, fs = 44e3, fc = 10e3, bw = 2e3, T = 0.01) :
 
    t = np.arange(0, T, 1/fs)  # Time vector
 
    # Preallocate the chirp signal array
    chirp_signal = np.zeros(len(data) * len(t))
 
    for i in range(len(data)):
        start_index = i * len(t)
        if data[i] == 0:
            chirp_signal[start_index:start_index + len(t)] = np.cos(2 * np.pi * ((fc + bw/2) * t + ((fc - bw/2) - ((fc + bw/2))) / (2 * T) * t**2))
        elif data[i] == 1:
            chirp_signal[start_index:start_index + len(t)] = np.cos(2 * np.pi * ((fc - bw/2) * t + ((fc + bw/2) - (fc - bw/2)) / (2 * T) * t**2))
 
    return chirp_signal
 
def find_start(signal):
    signal = np.array(signal)
 
    # Replace the above line with the following code
    variance_sign = []
    for i in range(50, len(signal), 10):
        actual_sign = np.sign(np.var(signal[i: i + 100]) - 0.25)
        variance_sign.append(actual_sign)
        if actual_sign == 1:
            break
 
   
    variance_sign = np.array(variance_sign)
    indice = np.where(np.diff(variance_sign) > 0)[0] + 11
    indice = 10 * indice
 
    # Check if indice is not empty
    if indice.size > 0:
        # Use the first valid index for slicing
        start_index = indice[0]  # Get the first index
        signal = signal[start_index:]  # Slice the signal from the first valid index
    else:
        print("No valid indices found.")
        return signal  # or handle as needed
 
    return signal
 
def chirp_demodulate(received_signal, fs, fc = 10e3, bw = 2e3, T = 0.01, debug = False):

    # normalize the signal amplitude by dividing by the 100th maximum value(added by mimmo02)-------------------------------------
    max = np.sort(np.abs(received_signal))[::-1][100]
    received_signal = received_signal / max
 
    received_signal = find_start(received_signal)
    print(len(received_signal))
 
    t = np.arange(0, T, 1/fs)  # Time vector for the chirp
 
    up = np.cos(2 * np.pi * ((fc + bw/2) * t + ((fc - bw/2) - ((fc + bw/2))) / (2 * T) * t**2))
    down = np.cos(2 * np.pi * ((fc - bw/2) * t + ((fc + bw/2) - (fc - bw/2)) / (2 * T) * t**2))
 
    result = []
 
    nyq = 0.5 * fs
    normal_cutoff = 0.5*fc / nyq
    b, a = butter(5, normal_cutoff, btype='low', analog=False)
 
 
 
    for i in range(int(len(received_signal)/len(t))) :
        signal_up = received_signal[i*len(t): (i + 1)*len(t)] * up
        signal_down = received_signal[i*len(t): (i + 1)*len(t)] * down
 
        signal_up = filtfilt(b, a, signal_up)
        signal_down = filtfilt(b, a, signal_down)
 
 
        mean_up = signal_up.sum() / len(signal_up)
        mean_down = signal_down.sum() / len(signal_down)
 
        var_up = np.var(signal_up)
        var_down = np.var(signal_down)
 
        if mean_up > mean_down and var_up < var_down/2 :
            result.append(0)
        elif mean_down > mean_up and var_down < var_up/2 :
            result.append(1)
 
       
        if debug :
            max = 1.5*np.max(np.abs(np.r_[received_signal[i*len(t): (i + 1)*len(t)], signal_up, signal_down]))
            fig, axs = plt.subplots(3)
            axs[0].plot(received_signal[i*len(t): (i + 1)*len(t)], "b")
            axs[0].set(ylim=[-max, max])
 
            axs[1].plot(signal_up, "r")
            axs[1].set_title(str("mean up :" + str(round(mean_up, 3))) + " / var up :" + str(round(np.var(signal_up), 3)))
            axs[1].set(ylim=[-max, max])
 
            axs[2].plot(signal_down, "g")
            axs[2].set_title(str("mean down :" + str(round(mean_down, 3)) + " / var down :" + str(round(np.var(signal_down), 3))))
            axs[2].set(ylim=[-max, max])
 
            plt.tight_layout()
            plt.show()
 
            if i == 0 :
                signal_up_all = signal_up
                signal_down_all = signal_down
            else :
                signal_up_all = np.concatenate((signal_up_all, signal_up))
                signal_down_all = np.concatenate((signal_down_all, signal_down))
 
    if debug :
        fig, axs = plt.subplots(3)
        axs[0].plot(received_signal, "b")
 
        axs[1].plot(signal_up_all, "r")
 
        axs[2].plot(signal_down_all, "g")
 
        plt.tight_layout()
        plt.show()
 
 
 
    return np.array(result)



import pyaudio
import numpy as np
import threading
import keyboard
import matplotlib.pyplot as plt

def record_audio(sample_rate=int(44e3)):
    """
    Record audio from the microphone until 'q' is pressed.

    Parameters:
    - sample_rate: The sample rate in Hz (default is 44e3).

    Returns:
    - fs: The sample rate used for recording.
    - audio_data: The recorded audio data as a NumPy array.
    """
    audio_data = []
    stop_event = threading.Event()

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, input=True, frames_per_buffer=1024)

    print("Recording... Press 'q' to stop.")

    while True:
        if keyboard.is_pressed('q'):
            stop_event.set()
            break
        data = np.frombuffer(stream.read(1024), dtype=np.int16)
        audio_data.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Convert the list to a NumPy array
    audio_data = np.concatenate(audio_data, axis=0)

    return sample_rate, audio_data

import IPython.display as ipd

def display_audio_signal(audio_signal, fs=44e3):
    ipd.display(ipd.Audio(audio_signal, rate=fs))