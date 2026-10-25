#!/bin/bash
#SBATCH --mem=32g
#SBATCH -c 4
#SBATCH --output=example.out

# inputs
in_fasta="/home/justas/fused_mpnn/make_msa/6tht.fa"
out_dir="/home/justas/fused_mpnn/make_msa/out_5"

# resources
CPU="4"
MEM="32"

# sequence databases
DB_UR30="/local/databases/uniclust/UniRef30_2021_06"
DB_BFD="/local/databases/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt"

if [ ! -s /local/databases/uniclust/UniRef30_2021_06_a3m.ffdata ]
then
    DB_UR30="/databases/uniclust/UniRef30_2021_06"
fi
if [ ! -s /local/databases/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt_a3m.ffdata ]
then
    DB_BFD="/databases/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt"
fi

trim_fasta="$in_fasta"

# setup hhblits command
export HHLIB=/home/robetta/rosetta_server_beta/src/hh-suite_20210726/hh-suite/build/
export PATH=$HHLIB/bin:$PATH
HHBLITS_UR30="hhblits -o /dev/null -mact 0.35 -maxfilt 100000000 -neffmax 20 -cov 25 -cpu $CPU -nodiff -realign_max 100000000 -maxseq 1000000 -maxmem $MEM -n 4 -d $DB_UR30"
HHBLITS_BFD="hhblits -o /dev/null -mact 0.35 -maxfilt 100000000 -neffmax 20 -cov 25 -cpu $CPU -nodiff -realign_max 100000000 -maxseq 1000000 -maxmem $MEM -n 4 -d $DB_BFD"

mkdir -p $out_dir/hhblits
tmp_dir="$out_dir/hhblits"
out_prefix="$out_dir/t000_"

# perform iterative searches against UniRef30
if [ ! -s ${out_prefix}.msa0.a3m ]
then
    prev_a3m="$trim_fasta"
    for e in 1e-10 1e-6 1e-3
    do
        echo "Running HHblits against UniRef30 with E-value cutoff $e"
        if [ ! -s $tmp_dir/t000_.$e.a3m ]
        then
            $HHBLITS_UR30 -i $prev_a3m -oa3m $tmp_dir/t000_.$e.a3m -e $e -v 0
        fi
        hhfilter -maxseq 100000 -id 95 -cov 75 -i $tmp_dir/t000_.$e.a3m -o $tmp_dir/t000_.$e.id90cov75.a3m
        hhfilter -maxseq 100000 -id 95 -cov 50 -i $tmp_dir/t000_.$e.a3m -o $tmp_dir/t000_.$e.id90cov50.a3m
        prev_a3m="$tmp_dir/t000_.$e.id90cov50.a3m"
        n75=`grep -c "^>" $tmp_dir/t000_.$e.id90cov75.a3m`
        n50=`grep -c "^>" $tmp_dir/t000_.$e.id90cov50.a3m`

        if ((n75>2000))
        then
            if [ ! -s ${out_prefix}.msa0.a3m ]
            then
                cp $tmp_dir/t000_.$e.id90cov75.a3m ${out_prefix}.msa0.a3m
                break
            fi
        elif ((n50>4000))
        then
            if [ ! -s ${out_prefix}.msa0.a3m ]
            then
                cp $tmp_dir/t000_.$e.id90cov50.a3m ${out_prefix}.msa0.a3m
                break
            fi
        else
            continue
        fi
    done

    # perform iterative searches against BFD if it failes to get enough sequences
    if [ ! -s ${out_prefix}.msa0.a3m ] 
    then
        e=1e-3
        echo "Running HHblits against BFD with E-value cutoff $e"
        if [ ! -s $tmp_dir/t000_.$e.bfd.a3m ]
        then
            $HHBLITS_BFD -i $prev_a3m -oa3m $tmp_dir/t000_.$e.bfd.a3m -e $e -v 0
        fi
        hhfilter -maxseq 100000 -id 95 -cov 75 -i $tmp_dir/t000_.$e.bfd.a3m -o $tmp_dir/t000_.$e.bfd.id90cov75.a3m
        hhfilter -maxseq 100000 -id 95 -cov 50 -i $tmp_dir/t000_.$e.bfd.a3m -o $tmp_dir/t000_.$e.bfd.id90cov50.a3m
        prev_a3m="$tmp_dir/t000_.$e.bfd.id90cov50.a3m"
        n75=`grep -c "^>" $tmp_dir/t000_.$e.bfd.id90cov75.a3m`
        n50=`grep -c "^>" $tmp_dir/t000_.$e.bfd.id90cov50.a3m`

        if ((n75>2000))
        then
            if [ ! -s ${out_prefix}.msa0.a3m ]
            then
                cp $tmp_dir/t000_.$e.bfd.id90cov75.a3m ${out_prefix}.msa0.a3m
            fi
        elif ((n50>4000))
        then
            if [ ! -s ${out_prefix}.msa0.a3m ]
            then
                cp $tmp_dir/t000_.$e.bfd.id90cov50.a3m ${out_prefix}.msa0.a3m
            fi
        fi
    fi

    if [ ! -s ${out_prefix}.msa0.a3m ]
    then
        cp $prev_a3m ${out_prefix}.msa0.a3m
    fi
fi
