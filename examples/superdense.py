#!/usr/bin/env python3

"""
Qwerty demo of superdense coding, a technique for transmitting two classical
bits using one qubit, as described in Section 2.3 of Nielsen and Chuang.
The two bits to send can be passed on the command line
(e.g., ``python superdense.py 01``).
"""

from argparse import ArgumentParser
from typing import Optional
from qwerty import *

def superdense_coding(payload: bit[2], acc: Optional[str] = None):
    bit0, bit1 = payload

    @qpu
    def kernel():
        alice, bob = '00' + '11'

        sent_to_bob = (
            alice | ({'0'>>'1', '1'>>'0'}
                     if bit0 else id)
                  | ('1' >> -'1'
                     if bit1 else id))

        return (sent_to_bob * bob
                | {'00' + '11', '00' + -'11',
                   '10' + '01', '01' + -'10'}
                  .measure)

    recovered_payload = kernel(acc=acc)
    return recovered_payload

if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('bits',
                        nargs='?',
                        default=None,
                        choices=['00', '01', '10', '11'],
                        help='The two bits to send. If not specified, the '
                             'protocol will be run four separate times for '
                             'every possible two-bit value.')
    parser.add_argument('--acc', '-a',
                        default=None,
                        help='Name of an accelerator. The default is local '
                             'simulation.')
    args = parser.parse_args()

    if args.bits is None:
        for i in range(1 << 2):
            payload = bit[2](i)
            print('Sent {} ==> Received {}'.format(
                payload, superdense_coding(payload, acc=args.acc)))
    else:
        payload = bit.from_str(args.bits)
        print('Sent {} ==> Received {}'.format(
            payload, superdense_coding(payload, acc=args.acc)))
