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
def huffman_encoding(textName):
    # Import text from file and print it
    text = open(textName+".txt", 'r').read()
    print('@source_coding_layer.HFM-> Text:', text)
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
    # Print the decoded text
    print('@source_decoding_layer.HFM-> Decoded text:', decoded_text)
    return decoded_text

# Encoding LZW algorithm
def LZW_encoding(textName):
    # Import text from file and print it
    text = open(textName+".txt", 'r').read()
    print('@source_coding_layer.LZW-> Text:', text)
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
    print('@source_coding_layer.LZW-> Dictionary size:', len(dictionary))
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
    # Print the decoded text
    print('@source_decoding_layer.LZW-> Decoded text:', text)
    return text

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
    if p.any():
        error_position = np.sum([2**i for i in range(m) if p[i] == 1])
        # flip the bit in the error position
        c[error_position-1] = 1 - c[error_position-1]
                
    data = np.dot(c, D) % 2
    return data

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

def channel_coding(MSG): 
    print("@channel_coding_layer-> ENCODING DATA ////////////////////////////////////////////////")
    packet_size = 11
    packets = split_packet(MSG, packet_size)
    codewords = []
    print("@channel_coding_layer-> ENCODING DATA PACKETS ////////////////////////////////////////")
    for packet in packets:
        codeword = channel_code_gen(packet, G_15_11)
        codewords.append(codeword)
        
    return codewords

def channel_decoding(COD):
    print("@channel_coding_layer-> DECODING DATA PACKETS ////////////////////////////////////////")
    packets = []
    for codeword in COD:
        packet = channel_decode_gen(codeword, H_15_11, D_15_11)
        packets.append(packet)
    print("@channel_coding_layer-> Decoded Packets: ", packets)
    # first packet is the number of zeros added
    fill_bits = packets.pop(0) 
    # binary to decimal
    fill_bits = int(''.join([str(x) for x in fill_bits]), 2) 
    # unite the packets
    packets = [item for sublist in packets for item in sublist]
    return packets,fill_bits

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