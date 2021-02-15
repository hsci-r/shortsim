import sys

from shortsim.align import align

def main():
    for line in sys.stdin:
        v1_id, v1_text, v2_id, v2_text = line.rstrip().split('\t')
        al = align(v1_text, v2_text, empty='-')
        al_str = ' '.join('{}:{}'.format(x,y) for (x, y, c) in al)
        n_id = sum((1-c) for (x,y,c) in al)
        d_lev = len(al)-n_id
        sim_nes = n_id/len(al)
        print(v1_id, v1_text, v2_id, v2_text, al_str, d_lev, sim_nes, sep='\t')

