import math
import random
from shor import order_finding

def shor(epsilon, num):
    if num % 2 == 0:
        return 2

    x = random.randint(2, num-1)
    if (y := math.gcd(x, num)) > 1:
        print('Got lucky! Skipping order subroutine')
        return y

    r = order_finding(epsilon, x, num)

    if r % 2 == 0 and pow(x, r//2, num) != -1:
        if (gcd := math.gcd(x**(r//2)-1, num)) > 1 \
                and gcd != num:
            return gcd
        if (gcd := math.gcd(x**(r//2)+1, num)) > 1 \
                and gcd != num:
            return gcd

    raise Exception("Shor's failed")

print(shor(0.2, 15))
