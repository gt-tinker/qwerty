from qwerty import *

@qpu
def teleport(payload: qubit) -> qubit:
    alice, bob = '00' + '11'

    bit_flip, sign_flip = (
        alice * payload | (flip if '_1' else id)
                        | (std * pm).measure)

    teleported_payload = (
        bob | (flip if bit_flip else id)
            | ('1' >> -'1'
               if sign_flip else id))

    return teleported_payload
