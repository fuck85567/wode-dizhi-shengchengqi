"""Optional CPU-only comparison; never called by normal program startup."""
import statistics
import time

from classifier_reference import classify_reference
from cpu_worker import classify_vanity
from test_vanity import fixtures


def main():
    config = dict(mode="wide")
    addresses = [a for a in fixtures() if classify_reference(a, config)] * 20
    for address in addresses:
        assert classify_vanity(address, config) == classify_reference(address, config)
    print("Synthetic candidate fixtures: {}; default 8-34, all rules".format(len(addresses)))
    rates = []
    for classify in (classify_reference, classify_vanity):
        samples = []
        for _ in range(5):
            started = time.perf_counter()
            for address in addresses:
                classify(address, config)
            samples.append(len(addresses)/(time.perf_counter()-started))
        rates.append(statistics.median(samples))
        print("{}: {:.0f} candidates/sec".format(classify.__name__, rates[-1]))
    print("CPU classification speedup: {:.2f}x".format(rates[1]/rates[0]))
    print("Single process; excludes cryptographic verification, IPC, disk and GPU.")


if __name__ == "__main__":
    main()
