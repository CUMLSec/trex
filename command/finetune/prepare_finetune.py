import os
from random import choice
import random
import argparse
import json
import gc


class Options():

    def __init__(self):
        self.initialized = False
        self.parser = None
        self.args = None
        self.epoch_needed_list = [-1]
        # now in the trace file, there is 4 epochs of traces ,now only need the one with '#######'
        # now only one epoch
        self.output_filename_prefix_list = ['train', 'valid']
        self.output_filename_prefix_prob_list = None
        self.output_filename_inter_list = ['static', 'inst_pos_emb', 'op_pos_emb', 'byte1', 'byte2', 'byte3', 'byte4',
                                           'arch_emb']
        self.positive_flag_list = [True, False]
        self.positive_flag_prob_list = [0.5, 0.5]
        self.archs = ['arm', 'mips', 'x86', 'x86_64']
        self.opts = ['O0', 'O1', 'O2', 'O3', 'orig',
                     'bcfobf', 'cffobf', 'splitobf', 'subobf', 'acdobf', 'indibran', 'strcry', 'funcwra']
        self.optimizations = ['O0', 'O1', 'O2', 'O3']
        self.obfs = ['bcfobf', 'cffobf', 'splitobf', 'subobf', 'acdobf', 'indibran', 'strcry', 'funcwra']
        self.arch_to_optListdict = {}
        self.opt_to_archListdict = {}

    def initialize(self, parser):

        '''
        parser.add_argument( '-archs','--archs_wanted_list', action="extend", nargs="*", type=str, required=False,
                             help="archs we want", default=archs)
        '''
        parser.add_argument('-n', '--sample_num', type=int, required=False, help="the number of samples", default=20000)
        parser.add_argument('-obf', '--only_obf', action='store_true', required=False, default=False)
        parser.add_argument('-optimization', '--only_optimization', action='store_true', required=False, default=False)
        parser.add_argument('-can_inter', '--valid_train_func_can_intersection_flag', action='store_true',
                            required=False,
                            help="the valid dataset and the training dataset can have  intersection", default=False)
        parser.add_argument('-c', '--train_test_ratio', type=float, required=False,
                            help="the ratio of training samples",
                            default=0.1)
        parser.add_argument('-newline', '--tokens_newline_number', type=int, required=False,
                            help="number of token new lines", default=512)
        parser.add_argument('-archs', '--archs_wanted_list', type=str, nargs='*', required=False,
                            help="archs wanted", default=self.archs)
        parser.add_argument('-opts', '--opts_wanted_list', type=str, nargs='*', required=False, default=self.opts)
        parser.add_argument('-arch_same', '--arch_must_same_flag', action='store_true', required=False, default=False)
        parser.add_argument('-opt_differ', '--opt_must_differ_flag', action='store_true', required=False, default=False)
        parser.add_argument('-opt_same', '--opt_must_same_flag', action='store_true', required=False, default=False)
        parser.add_argument('-i', '--functraces_folder_path', type=str, required=False, default='data-raw/functraces')
        parser.add_argument('-o', '--output_folder_path', type=str, required=False, default='data-src/similarity')
        parser.add_argument('-trunc', '--tokens_truncate_flag', action='store_true', required=False, default=False)
        parser.add_argument('-minlen', '--trace_min_len', type=int, required=False, default=50)
        # if the value is 10, then positive and negative training data 1:9, and these 9 negative pairs share the same
        # left function with positive pair
        parser.add_argument('-training_cycle', '--training_cycle', type=int, required=False, default=10)
        parser.add_argument('-valid_cycle', '--valid_cycle', type=int, required=False, default=10)
        parser.add_argument('-bins', '--binfile_keyword_list_needed', type=str, nargs='*',
                            required=False, default=[])

        self.initialized = True
        self.parser = parser
        self.banned_function_list = ['skip_white', 'free_dir', 'parse_name', 'blake2b_increment_counter', 'main',
                                     'blake2b_compress', 'register_tm_clones', 'write_pending', 'millerrabin',
                                     'blake2b_set_lastblock', 'print_name', 'read_string', 'blake2b_init_param',
                                     'check', 'base64_decode', 'print_stuff', 'blake2b_init0', 'usage',
                                     'cleanup', 'frame_dummy', 'print_entry', 'print_user', 'print_stats',
                                     'base64_encode', 'base64url_encode', 'blake2b_init', 'deregister_tm_clones']

    def parse(self):

        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser()
            self.initialize(parser)

        self.args = self.parser.parse_args()
        self.output_filename_prefix_prob_list = [self.args.train_test_ratio, 1 - self.args.train_test_ratio]

        for arch in self.archs:
            self.arch_to_optListdict[arch] = []

        for opt in self.opts:
            self.opt_to_archListdict[opt] = []

        for arch_opt_name in os.listdir(self.args.functraces_folder_path):
            tmp_list = arch_opt_name.split('-')
            tmp_arch = arch_str_to_arch_dict[f'{tmp_list[0]}-{tmp_list[1]}']
            tmp_opt = f'{tmp_list[2]}'
            self.arch_to_optListdict[tmp_arch].append(tmp_opt)
            self.opt_to_archListdict[tmp_opt].append(tmp_arch)

        return self.args


class BiaryFileInfo(object):

    def __init__(self, arch=None, opt=None, func_name=None, proj_name=None, trace_path=None):
        """

        :param archs_opt_name: string, like 'x86-32-O3' or 'mips-32-O1'
                it would be transformed to the variable member of the object
        """

        self.arch = arch
        self.opt = opt
        self.func_name = func_name
        self.proj_name = proj_name
        self.trace_path = trace_path

    def __str__(self):
        return f'BiaryFileInfo:trace_file_info {self.trace_path}, arch {self.arch}, ' \
               f'opt {self.opt}, func_name {self.func_name}, proj_name {self.proj_name}'


def two_lists_intersection(list1, list2):
    """

    :param list1:
    :param list2:
    :return: list, intersection of lists
    """
    return list(set(list1).intersection(set(list2)))


def list1_minus_list2(list1, list2):
    return list(set(list1).difference(set(list2)))


# def get_two_random_func_names(arch_opt_folder_path1, arch_opt_folder_path2, positive_flag = False, options=None):
def get_two_random_func_names(arch_opt_folder_path1, arch_opt_folder_path2, positive_flag=False,
                              last_training_function_1=None, last_valid_function_1=None,
                              dl_dataset_cat=None, options=None):
    func_name_list1 = os.listdir(arch_opt_folder_path1)
    func_name_list2 = os.listdir(arch_opt_folder_path2)
    # pick two funcs
    if positive_flag:
        # must be the same function and the same version binary
        # now allow two totally same file exist
        # assert arch_opt_name1 != arch_opt_name2

        # for positive
        func_name_intersection_list = two_lists_intersection(func_name_list1, func_name_list2)
        if len(func_name_intersection_list) < 1:
            return None, None
        random_func_name = choice(func_name_intersection_list)
        random_func_name1 = random_func_name
        random_func_name2 = random_func_name


    else:  # negative

        if len(func_name_list1) == 0:
            return None, None

        assert len(func_name_list1) > 0
        # the negative sample is guaranteed to be
        if dl_dataset_cat == 'train' and last_training_function_1 is not None:
            random_func_name1 = last_training_function_1
            if random_func_name1 not in func_name_list1:
                return None, None
        elif dl_dataset_cat == 'valid' and last_valid_function_1 is not None:
            random_func_name1 = last_valid_function_1
            if random_func_name1 not in func_name_list1:
                return None, None
        else:
            random_func_name1 = choice(func_name_list1)
        # random_func_name1 = choice(func_name_list1)
        random_func_name2_pool = list1_minus_list2(func_name_list2, [random_func_name1])
        if len(random_func_name2_pool) == 0:
            return None, None

        assert len(random_func_name2_pool) > 0

        random_func_name2 = choice(random_func_name2_pool)

        assert random_func_name2 != random_func_name1

    assert random_func_name1 in func_name_list1
    assert random_func_name2 in func_name_list2
    # print("~~~~"," ", random_func_name1, " ", random_func_name2)
    if random_func_name1 in options.banned_function_list or random_func_name2 in options.banned_function_list:
        return None, None
    return random_func_name1, random_func_name2


def get_two_random_trace_file_info(func_folder_path1, func_folder_path2, positive_flag=False, options=None):
    trace_name_list1 = os.listdir(func_folder_path1)
    trace_name_list2 = os.listdir(func_folder_path2)
    # print(trace_name_list1)
    # print(trace_name_list2)
    if len(options.args.binfile_keyword_list_needed) > 0:
        trace_name_list1 = list(set([bin_name for bin_name in trace_name_list1
                                     for keyword in options.args.binfile_keyword_list_needed if keyword in bin_name]))
        trace_name_list2 = list(set([bin_name for bin_name in trace_name_list2
                                     for keyword in options.args.binfile_keyword_list_needed if keyword in bin_name]))
    if len(trace_name_list1) == 0 or len(trace_name_list2) == 0:
        return None, None, None, None

    if positive_flag:

        trace_name_intersection_list = two_lists_intersection(trace_name_list1, trace_name_list2)
        if len(trace_name_intersection_list) < 1:
            return None, None, None, None

        trace_name = choice(trace_name_intersection_list)
        random_trace_name1 = trace_name
        random_trace_name2 = trace_name

        # return trace_path1, trace_path2

    else:  # need test
        """
        Different functions, other variables are random(Fixing arch or opt now only works for positive sample)
        """
        # They must be different functions

        if len(trace_name_list1) == 0 or len(trace_name_list2) == 0:
            return None, None, None, None
        random_trace_name1 = choice(trace_name_list1)
        random_trace_name2 = choice(trace_name_list2)

    trace_path1 = f'{func_folder_path1}/{random_trace_name1}'
    trace_path2 = f'{func_folder_path2}/{random_trace_name2}'

    assert os.path.isfile(trace_path1)
    assert os.path.isfile(trace_path2)

    return random_trace_name1, random_trace_name2, trace_path1, trace_path2


# def pick_two_trace_path(arch1, opt1, arch2, opt2, positive_flag = False, options=None):
def pick_two_trace_path(arch1, opt1, arch2, opt2, positive_flag=False, last_training_function_1=None,
                        last_valid_function_1=None, dl_dataset_cat=None, options=None):
    # print(arch1, opt1, arch2, opt2, positive_flag)
    # if arch in archs_wanted_list and opt in opts:
    #    return None, None
    trace_file_info1 = None
    trace_file_info2 = None

    assert arch1 in options.args.archs_wanted_list
    assert arch2 in options.args.archs_wanted_list
    assert opt1 in options.args.opts_wanted_list
    assert opt2 in options.args.opts_wanted_list

    # print(arch1, opt1, arch2, opt2, positive_flag)
    arch_to_optListdict = options.arch_to_optListdict
    opt_to_archListdict = options.opt_to_archListdict
    if opt1 not in arch_to_optListdict[arch1] or opt2 not in arch_to_optListdict[arch2]:
        return None, None

    assert opt1 in arch_to_optListdict[arch1]
    assert opt2 in arch_to_optListdict[arch2]
    assert arch1 in opt_to_archListdict[opt1]
    assert arch2 in opt_to_archListdict[opt2]

    # if positive_flag:
    if options.args.arch_must_same_flag:
        if arch1 != arch2:
            return None, None
    if options.args.opt_must_differ_flag:
        if opt1 == opt2:
            return None, None
    if options.args.opt_must_same_flag:
        if opt1 != opt2:
            return None, None

    arch_opt_name1 = f'{arch_to_arch_str_dict[arch1]}-{opt1}'
    arch_opt_name2 = f'{arch_to_arch_str_dict[arch2]}-{opt2}'
    # it should be 'functraces/arm-32-O3' or 'functraces/mips-32-O1' .etc
    arch_opt_folder_path1 = f'{options.args.functraces_folder_path}/{arch_opt_name1}'
    arch_opt_folder_path2 = f'{options.args.functraces_folder_path}/{arch_opt_name2}'

    print(arch_opt_folder_path1)
    assert os.path.isdir(arch_opt_folder_path1)
    assert os.path.isdir(arch_opt_folder_path2)

    random_func_name1, random_func_name2 = \
        get_two_random_func_names(arch_opt_folder_path1, arch_opt_folder_path2, positive_flag, last_training_function_1,
                                  last_valid_function_1, dl_dataset_cat, options)

    if random_func_name1 is None or random_func_name2 is None:
        return None, None

    # func_folder_path should be like 'functraces/x86-32-acdobf/sha1_stream'
    func_folder_path1 = f'{arch_opt_folder_path1}/{random_func_name1}'
    func_folder_path2 = f'{arch_opt_folder_path2}/{random_func_name2}'
    # print(func_folder_path1)
    # print(func_folder_path2)

    if not os.path.isdir(func_folder_path1):
        return None, None

    assert os.path.isdir(func_folder_path1)
    assert os.path.isdir(func_folder_path2)

    random_trace_name1, random_trace_name2, trace_path1, trace_path2 = \
        get_two_random_trace_file_info(func_folder_path1, func_folder_path2, positive_flag, options)
    # print(random_trace_name1)
    # print(random_trace_name2)
    if trace_path1 is None or trace_path2 is None:
        return None, None

    trace_file_info1 = BiaryFileInfo(arch1, opt1, random_func_name1, random_trace_name1, trace_path1)
    trace_file_info2 = BiaryFileInfo(arch2, opt2, random_func_name2, random_trace_name2, trace_path2)

    return trace_file_info1, trace_file_info2


def value_list2seq(value_list):
    """

    :param value_list: like ['########', '00510000', '########', '00510000' ...
    :return: like [['##', '##', '##', '##'], ['00', '51', '00', '00'], ...
    """
    # print(value_list)
    value_list_transpose = []
    for value in value_list:
        # print(len(value))
        value_list_transpose.append([value[i:i + 2] for i in range(0, len(value), 2)])
    return value_list_transpose


def value_to_four_byte_list(value_list):
    byte_sequence_list = value_list2seq(value_list)
    # print(byte_sequence_list)
    return [i[0] for i in byte_sequence_list], [i[1] for i in byte_sequence_list] \
        , [i[2] for i in byte_sequence_list], [i[3] for i in byte_sequence_list]


def trace_file_info_to_data_list(trace_file_info, options=None):
    """

    :param trace_file_info:
    :return: like [[code],[inst_emb],[inst_pos_emb],[arch_id],[byte1],[byte2],[byte3],[byte4],
                [arch,opt/obfuscation,proj_name,func_name]]
    """

    assert len(options.epoch_needed_list) == 1
    assert os.path.isfile(trace_file_info.trace_path)
    data_list = []
    # print(trace_file_info.trace_path)
    size = os.path.getsize(trace_file_info.trace_path)
    if size == 0:
        return None
    with open(trace_file_info.trace_path, 'r') as trace:
        trace_epoch_index = options.epoch_needed_list[0]
        # if the file is empty, error will happen

        trace_list = json.load(trace)
        # length is 4, index is 3 yes, index is 4 no
        if len(trace_list) <= trace_epoch_index:
            return None
        trace_epoch_list = trace_list[trace_epoch_index]
        data_list.append(trace_epoch_list[0])
        data_list.append(trace_epoch_list[2])
        data_list.append(trace_epoch_list[3])
        bytes_list = list(value_to_four_byte_list(trace_epoch_list[1]))
        data_list.append(bytes_list[0])
        data_list.append(bytes_list[1])
        data_list.append(bytes_list[2])
        data_list.append(bytes_list[3])
        data_list.append([arch_to_arch_data_dict[trace_file_info.arch], trace_file_info.opt,
                          trace_file_info.proj_name, trace_file_info.func_name])

    return data_list


def write_str_to_file(output_filename_prefix, output_filename_inter, output_filename_suffix,
                      str_to_write, write_type, options):
    final_data_folder = options.args.output_folder_path
    if output_filename_suffix == "label":
        filename = f'{output_filename_prefix}.label'
    else:
        filename = f'{output_filename_prefix}.{output_filename_inter}.{output_filename_suffix}'
    file_path = os.path.join(final_data_folder, filename)
    with open(file_path, write_type) as wf:
        wf.write(str_to_write)


def append_dict_to_outputfile_as_str(pair_dict, output_filename_prefix, options=None):
    # output_filename_prefix = random.choices(options.output_filename_prefix_list, output_filename_prefix_prob_list)[0]
    # final_data_folder = options.args.output_folder_path
    # print(output_filename_prefix)
    f1_list = pair_dict['f1']  # input0
    f2_list = pair_dict['f2']  # input1
    output_filename_inter_list = options.output_filename_inter_list
    assert len(f1_list) == len(output_filename_inter_list)
    assert len(f2_list) == len(output_filename_inter_list)
    tokens_newline_number = options.args.tokens_newline_number
    tokens_truncate_flag = options.args.tokens_truncate_flag
    if not tokens_truncate_flag and len(f1_list[0]) > tokens_newline_number:
        print("f1 too long, discard")
        return None
    if not tokens_truncate_flag and len(f2_list[0]) > tokens_newline_number:
        print("f2 too long, discard")
        return None
    line_token_num1 = min(len(f1_list[0]), tokens_newline_number)
    line_token_num2 = min(len(f2_list[0]), tokens_newline_number)
    if tokens_newline_number < len(f1_list[0]):
        print("truncate")
    if line_token_num1 == 0 or line_token_num2 == 0:
        return None

    if line_token_num1 < options.args.trace_min_len or line_token_num2 < options.args.trace_min_len:
        print("Trace too short !! The length of two traces are", str(line_token_num1), " ", str(line_token_num2))
        return None

    assert line_token_num1 > 0 and line_token_num1 <= tokens_newline_number
    assert line_token_num2 > 0 and line_token_num2 <= tokens_newline_number

    if tokens_truncate_flag:
        assert line_token_num1 != None
        f1_list = [arr[:line_token_num1] for arr in f1_list]
        f2_list = [arr[:line_token_num2] for arr in f2_list]

    global train_pair_count, valid_pair_count
    if output_filename_prefix == "train":
        train_pair_count += 1
    elif output_filename_prefix == "valid":
        valid_pair_count += 1

    print("----------------------------------------------")

    for output_filename_inter_index in range(len(output_filename_inter_list) - 1):
        # write ['train', 'valid'] ['static', 'inst_emb', 'inst_pos_emb', 'byte1', 'byte2', 'byte3', 'byte4']
        # [input0 input1]
        output_filename_inter = output_filename_inter_list[output_filename_inter_index]
        assert line_token_num1 is not None
        assert len(f1_list[0]) == line_token_num1 and len(f2_list[0]) == line_token_num2
        # print(f1_list[output_filename_inter_index])
        # print(f1_list[output_filename_inter_index])
        str_to_write1 = " ".join([str(x) for x in f1_list[output_filename_inter_index]])
        str_to_write2 = " ".join([str(x) for x in f2_list[output_filename_inter_index]])
        write_str_to_file(output_filename_prefix, output_filename_inter, 'input0', str_to_write1 + "\n", 'a', options)
        write_str_to_file(output_filename_prefix, output_filename_inter, 'input1', str_to_write2 + "\n", 'a', options)

    output_filename_inter = output_filename_inter_list[len(output_filename_inter_list) - 1]  # 'arch_emb'
    arch1 = f1_list[len(output_filename_inter_list) - 1][0]  # 1st element in like ["x64", "O2", "", ""]
    arch2 = f2_list[len(output_filename_inter_list) - 1][0]

    append_arch_list1 = [arch1 for i in range(line_token_num1)]
    append_arch_list2 = [arch2 for i in range(line_token_num2)]
    assert append_arch_list1 is not None
    str_to_write1 = " ".join(append_arch_list1)
    str_to_write2 = " ".join(append_arch_list2)
    write_str_to_file(output_filename_prefix, output_filename_inter, 'input0', str_to_write1 + "\n", 'a', options)
    write_str_to_file(output_filename_prefix, output_filename_inter, 'input1', str_to_write2 + "\n", 'a', options)

    # label
    str_to_write = str(pair_dict['label'])
    assert str_to_write == "1" or str_to_write == "-1"
    global positive_count, negative_count
    if str_to_write == "1":
        positive_count += 1
    elif str_to_write == "-1":
        negative_count += 1
    print("the label written:", str_to_write)
    write_str_to_file(output_filename_prefix, None, 'label', str_to_write + "\n", 'a', options)

    return True


def get_pair_as_sample(trace_file_info1, trace_file_info2, positive_flag=False,
                       pick_success_count=None, dl_dataset_cat=None, options=None):
    # trace_path1 = trace_file_info1.trace_path
    # trace_path2 = trace_file_info2.trace_path

    assert len(options.epoch_needed_list) == 1  # now only need 1 with trace #####
    assert pick_success_count is not None and dl_dataset_cat is not None
    # dl_dataset_cat = random.choices(options.output_filename_prefix_list, output_filename_prefix_prob_list)[0]
    assert dl_dataset_cat == "train" or dl_dataset_cat == "valid"
    if dl_dataset_cat == "train":
        other_dl_dataset_cat = "valid"
    elif dl_dataset_cat == "valid":
        other_dl_dataset_cat = "train"
    global dl_dataset_cat_to_func_set_dict
    valid_train_func_no_intersection_flag = not options.args.valid_train_func_can_intersection_flag
    if valid_train_func_no_intersection_flag:
        if trace_file_info1.func_name in dl_dataset_cat_to_func_set_dict[other_dl_dataset_cat]:
            return None
        if trace_file_info2.func_name in dl_dataset_cat_to_func_set_dict[other_dl_dataset_cat]:
            return None
    result_dict = {}
    data_list1 = trace_file_info_to_data_list(trace_file_info1, options)
    if data_list1 is None:
        return None
    data_list2 = trace_file_info_to_data_list(trace_file_info2, options)
    if data_list2 is None:
        return None

    result_dict['f1'] = data_list1
    result_dict['f2'] = data_list2

    if positive_flag:
        result_dict['label'] = 1
    else:
        result_dict['label'] = -1

    assert os.path.isdir(options.args.output_folder_path)

    '''
    if positive_flag:
        with open(f'{positive_output_folder_path}/{str(pick_success_count)}', 'w') as wf:
            wf.write(json.dumps(result_dict))
    else:
        with open(f'{negative_output_folder_path}/{str(pick_success_count)}', 'w') as wf:
            wf.write(json.dumps(result_dict))    
    '''
    print("Func name :", trace_file_info1.func_name, " and ", trace_file_info2.func_name)
    dl_dataset_cat_to_func_set_dict[dl_dataset_cat].add(trace_file_info1.func_name)
    dl_dataset_cat_to_func_set_dict[dl_dataset_cat].add(trace_file_info2.func_name)

    if append_dict_to_outputfile_as_str(result_dict, dl_dataset_cat, options) is None:
        return None
    return True


def collect_pairs(options):
    """

    :param
    :return: collect pairs, each pair as a dict in a intermediate file
    """
    pick_success_count = 0
    train_success_count = 0
    valid_success_count = 0
    # positive_flag_keep_same = False
    last_pick_success_count = -1
    last_positive_flag = True
    last_dl_dataset_cat = "valid"
    last_training_function1 = None
    last_valid_function1 = None
    while True:
        if pick_success_count >= options.args.sample_num:
            break

        # if pick_success_count % print_cycle == 0:
        #    print(f'Now have collected {pick_success_count}')
        random_arch1 = choice(options.args.archs_wanted_list)
        random_opt1 = choice(options.args.opts_wanted_list)
        random_arch2 = choice(options.args.archs_wanted_list)
        random_opt2 = choice(options.args.opts_wanted_list)

        if pick_success_count == last_pick_success_count:  # keep the choice in the last loop
            positive_flag = last_positive_flag
            dl_dataset_cat = last_dl_dataset_cat
        else:
            # positive_flag = random.choices(options.positive_flag_list, options.positive_flag_prob_list)[0]
            dl_dataset_cat = random.choices(options.output_filename_prefix_list
                                            , options.output_filename_prefix_prob_list)[0]
            if dl_dataset_cat == 'train':
                if train_success_count % options.args.training_cycle == 0:
                    positive_flag = True
                else:
                    positive_flag = False
            else:
                assert dl_dataset_cat == 'valid'

                if valid_success_count % options.args.valid_cycle == 0:
                    positive_flag = True
                else:
                    positive_flag = False

        print(pick_success_count, " ", positive_flag, " cat:", dl_dataset_cat, " last_cat:", last_dl_dataset_cat)

        last_pick_success_count = pick_success_count
        last_positive_flag = positive_flag
        last_dl_dataset_cat = dl_dataset_cat
        # print(random_arch1, random_opt1, random_arch2, random_opt2, positive_flag)
        trace_file_info1, trace_file_info2 = \
            pick_two_trace_path(random_arch1, random_opt1, random_arch2, random_opt2, positive_flag,
                                last_training_function1, last_valid_function1, dl_dataset_cat, options)
        '''
        if positive_flag:
            pair_file_count = pick_success_count
        else:
            pair_file_count = pick_success_count + options.args.sample_num
        '''
        pair_file_count = pick_success_count
        if trace_file_info1 is not None and trace_file_info2 is not None:
            # print("-----", trace_file_info1, "\n", trace_file_info2)
            tmp_flag = get_pair_as_sample(trace_file_info1, trace_file_info2, positive_flag,
                                          pair_file_count, dl_dataset_cat, options)
            if tmp_flag is None:
                # positive_flag_keep_same = True
                continue

            pick_success_count += 1
            if dl_dataset_cat == "train":
                last_training_function1 = trace_file_info1.func_name
                train_success_count += 1
            elif dl_dataset_cat == "valid":
                last_valid_function1 = trace_file_info1.func_name
                valid_success_count += 1

            print("Successfully write the ", str(pick_success_count), " pair")
            print("trace1's path is ", trace_file_info1.trace_path)
            print("trace2's path is ", trace_file_info2.trace_path)
            print("positive_flag: ", positive_flag)
            if dl_dataset_cat == "train":
                tmp_ct = train_pair_count
            else:
                tmp_ct = valid_pair_count
            print("This is the ", tmp_ct, " th ", dl_dataset_cat)

            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            # print(dl_dataset_cat_to_func_set_dict)
            # positive_flag_keep_same = False
        # else:
        # positive_flag_keep_same = True


def init_output_files(options):
    # create the output files
    # final_data_folder = options.args.output_folder_path
    # final_data_names_list = os.listdir(final_data_folder)
    # assert len(final_data_names_list) == 0  # should be cleared before
    for output_filename_prefix in options.output_filename_prefix_list:
        for output_filename_inter in options.output_filename_inter_list:
            write_str_to_file(output_filename_prefix, output_filename_inter, 'input0', "", 'w', options)
            write_str_to_file(output_filename_prefix, output_filename_inter, 'input1', "", 'w', options)
            # assert not os.path.isfile(file_path0)
            # assert not os.path.isfile(file_path1)

        write_str_to_file(output_filename_prefix, None, 'label', "", 'w', options)
        # assert not os.path.isfile(filename)


arch_str_to_arch_dict = {'x86-32': 'x86',
                         'x86-64': 'x86_64',
                         'arm-32': 'arm',
                         # 'arm-64': Cs(CS_ARCH_ARM64, CS_MODE_ARM + CS_MODE_LITTLE_ENDIAN),
                         'mips-32': 'mips'}
# 'mips-64': Cs(CS_ARCH_MIPS, CS_MODE_MIPS64 + CS_MODE_BIG_ENDIAN)}

arch_to_arch_str_dict = {'x86': 'x86-32',
                         'x86_64': 'x86-64',
                         'arm': 'arm-32',
                         # 'arm-64': Cs(CS_ARCH_ARM64, CS_MODE_ARM + CS_MODE_LITTLE_ENDIAN),
                         'mips': 'mips-32'}

arch_to_arch_data_dict = {'x86': 'x86',
                          'x86_64': 'x64',
                          'arm': 'arm',
                          # 'arm-64': Cs(CS_ARCH_ARM64, CS_MODE_ARM + CS_MODE_LITTLE_ENDIAN),
                          'mips': 'mips'}

# some count number
train_pair_count = 0
valid_pair_count = 0
positive_count = 0
negative_count = 0

# Now there should be nothing in the folder
# assert len(os.listdir(final_data_folder)) == 0


# func_name_set = set()
# funcName_to_archOptList_dict = {}


# print(arch_to_optListdict)
# print(opt_to_archListdict)

if __name__ == '__main__':

    # parser
    optionss = Options()
    args = optionss.parse()
    if args.only_optimization:
        args.opts_wanted_list = optionss.optimizations
    if args.only_obf:
        args.opts_wanted_list = optionss.obfs
    # test argparse
    '''
    
    print("sample number: ", optionss.args.sample_num)
    for i in range(optionss.args.sample_num):
        pass
    print(valid_train_func_no_intersection_flag)
    print(optionss.args.archs_wanted_list)
    print(optionss.args.opts_wanted_list)
    print(optionss.args.arch_must_same_flag, " ", optionss.args.opt_must_differ_flag)  
    '''

    assert (set(optionss.args.archs_wanted_list) <= set(optionss.archs))
    assert (set(optionss.args.opts_wanted_list) <= set(optionss.opts))

    dl_dataset_cat_to_func_set_dict = {}
    for dl_dataset_cat in optionss.output_filename_prefix_list:
        dl_dataset_cat_to_func_set_dict[dl_dataset_cat] = set()

    # final_data_folder = f'{optionss.args.output_folder_path}/final_data_ver2'
    final_data_folderr = optionss.args.output_folder_path

    if not os.path.isdir(final_data_folderr):
        os.mkdir(final_data_folderr)

    init_output_files(optionss)
    collect_pairs(optionss)
    print("train pair count: ", train_pair_count, " valid_pair_count: ",
          valid_pair_count, "positive_count: ", positive_count, "negative_count: ", negative_count)

    valid_train_func_no_intersection_flagg = not args.valid_train_func_can_intersection_flag
    if valid_train_func_no_intersection_flagg:
        assert len(two_lists_intersection(list(dl_dataset_cat_to_func_set_dict["train"]),
                                          list(dl_dataset_cat_to_func_set_dict["valid"]))) == 0

'''
a = [1,2,3,4]
b = [4,5,8,9]
c = [5,6,7,8]
print(two_lists_intersection(a,b), " ", two_lists_intersection(b,c), " ", two_lists_intersection(c, a))
print(list1_minus_list2(a,b), " ", list1_minus_list2(b,c), " ", list1_minus_list2(c, a))


tmp = value_to_four_byte_list(['########', '00510000', '########', '00510000', '########', '0060fff8', '########', '########', '########', '########', '0060fff8', '########', '00000004', '########', '########', '24aca3b1', '########', '286f4105', '########', '00000002', '########', '00000002', '########', '########', '########', '########', '0060fff8', '########', '00000004', '########', '########', '########', '0d448197', '########', '########', 'db535c51', '########', '00000008', '########', '########', 'db535c51', '########', '########', '00010019', '########', '0026359c', '########', '########', '002735bc', '########', '########', '########', '########', 'da9ae288', '########', '002735bc', '########', '########', '0060fff8', '########'])
print(tmp)
'''

# garbage collect
