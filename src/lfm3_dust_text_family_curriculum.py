"""Frozen *task-family-disjoint* public text curriculum for LFM3 Dust.

Train the derivative-free attention head on FIRST/LAST word extraction,
then freeze it before scoring its predictions of gains on UPPERCASE rewriting.
The transfer split's measured gain labels are *evaluation data* and never
enter the attention memory, head tuning, shrinkage, or optimizer decisions.
No actual selection/final-test partition is accessed or opened.

This is deliberately a novel family vs the earlier arithmetic procedures,
but still synthetic public text; it is NOT a real-world generalization test.
"""
from __future__ import annotations
import hashlib
import json
import random

SCHEMA="sensorgraft.lfm3.dust-disjoint-text-families.v1"
TRAIN_EPISODES=12
TRANSFER_EPISODES=6
SEEDS=(10,11,12)
WORDS=("amber","birch","cedar","delta","ember","fern","grove","harbor",
       "iris","juniper","kestrel","linden","marble","nectar","olive",
       "pebble","quartz","river","spruce","thistle","umber","violet",
       "willow","zephyr","acorn","breeze","clover","dune","elm","frost")


def _one(family,first,last):
    if family=="first_word":
        prompt=f"Follow the instruction exactly. Return only the FIRST word. Words: {first} {last}\nAnswer:"
        answer=" "+first
        foil=" "+last
    elif family=="last_word":
        prompt=f"Follow the instruction exactly. Return only the LAST word. Words: {first} {last}\nAnswer:"
        answer=" "+last
        foil=" "+first
    elif family=="uppercase":
        prompt=f"Follow the instruction exactly. Convert the word to UPPERCASE. Word: {first}\nAnswer:"
        answer=" "+first.upper()
        foil=" "+first
    else:
        raise ValueError("unknown_task_family")
    return {"family":family,"prompt":prompt,"answer":answer,"foil":foil,
            "input_key":f"{family}|{first}|{last}"}


def curriculum(seed):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("unregistered_transfer_seed")
    rng=random.Random(500071+seed)
    train,transfer=[],[]
    seen=set()
    for i in range(TRAIN_EPISODES+TRANSFER_EPISODES):
        family=("first_word" if i%2==0 else "last_word") if i<TRAIN_EPISODES else "uppercase"
        for _ in range(500):
            first,last=rng.sample(WORDS,2)
            row=_one(family,first,last)
            # Distinct example input, including against any possible
            # same-family duplicate within the same seed.
            if row["input_key"] not in seen:
                seen.add(row["input_key"])
                (train if i<TRAIN_EPISODES else transfer).append(row)
                break
        else:
            raise RuntimeError("no_unique_task_example")
    assert len(train)==TRAIN_EPISODES and len(transfer)==TRANSFER_EPISODES
    assert {r["family"] for r in train}=={"first_word","last_word"}
    assert {r["family"] for r in transfer}=={"uppercase"}
    return {"train":train,"transfer_evaluation":transfer}


def digest(rows):
    public=[{k:v for k,v in row.items() if k!="input_key"} for row in rows]
    return hashlib.sha256(json.dumps(public,sort_keys=True,
                                      separators=(",",":"),allow_nan=False).encode()).hexdigest()


def manifest(seed):
    return {part:digest(rows) for part,rows in curriculum(seed).items()}
