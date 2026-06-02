# EV013: Paper page - AgentClinic: a multimodal agent benchmark to evaluate AI ...

URL: https://huggingface.co/papers/2405.07960

Source: openclaw_web_search

Year: unknown

Accessed: 2026-06-02T05:45:17.748Z

## Abstract Or Metadata

Join the discussion on this paper page AgentClinic: a multimodal agent benchmark to evaluate AI in simulated clinical environments

## Fetched Snapshot (via OpenClaw infer web fetch)



Paper page - AgentClinic: a multimodal agent benchmark to evaluate AI in simulated

clinical environments

Hugging Face

Models

Datasets

Spaces

Buckets

new

Docs

Enterprise

Pricing

Website

Tasks

HuggingChat

Collections

Languages

Organizations

Community

Blog

Posts

Daily Papers

Learn

Discord

Forum

GitHub

Solutions

Team &amp; Enterprise

Hugging Face PRO

Enterprise Support

Inference Providers

Inference Endpoints

Storage Buckets

Log In

Sign Up

Papers

arxiv:2405.07960

Copy markdown

AgentClinic: a multimodal agent benchmark to evaluate AI in simulated

clinical environments

Published on May 13, 2024

Upvote  1

Authors:

Samuel Schmidgall

,

Rojin Ziaei

,

Carl Harris

,

Eduardo Reis

,

Jeffrey Jopling

,

Michael Moor

Abstract

AgentClinic evaluates the interactive decision-making abilities of LLMs in simulated clinical environments, showing that cognitive biases significantly impact diagnostic accuracy and patient compliance.

AI-generated summary

Diagnosing and managing a patient is a complex, sequential decision making
process that requires physicians to obtain information -- such as which tests
to perform -- and to act upon it. Recent advances in artificial intelligence
(AI) and

large language models

(

LLMs

) promise to profoundly impact clinical
care. However, current evaluation schemes overrely on static medical
question-answering benchmarks, falling short on interactive decision-making
that is required in real-life clinical work. Here, we present

AgentClinic

: a

multimodal benchmark

to evaluate

LLMs

in their ability to operate as agents in
simulated clinical environments. In our benchmark, the doctor agent must
uncover the patient's diagnosis through

dialogue

and

active data collection

. We
present two open medical agent benchmarks: a multimodal image and

dialogue

environment,

AgentClinic

-NEJM, and a

dialogue

-only environment,

AgentClinic

-

MedQA

. We embed cognitive and implicit biases both in patient and
doctor agents to emulate realistic interactions between biased agents. We find
that introducing bias leads to large reductions in

diagnostic accuracy

of the
doctor agents, as well as reduced

compliance

, confidence, and follow-up
consultation willingness in patient agents. Evaluating a suite of
state-of-the-art

LLMs

, we find that several models that excel in benchmarks
like

MedQA

are performing poorly in

AgentClinic

-

MedQA

. We find that the LLM
used in the patient agent is an important factor for performance in the

AgentClinic

benchmark. We show that both having limited interactions as well as
too many interaction reduces

diagnostic accuracy

in doctor agents. The code and
data for this work is publicly available at https://

AgentClinic

.github.io.

View arXiv page

View PDF

Add to collection

Community

Edit  Preview

Upload images, audio, and videos by dragging in the text input, pasting, or  clicking here .

Tap or paste here to upload images

Comment

·

Sign up  or  log in  to comment

Upvote  1

Get this paper in your agent:

hf papers read 2405.07960

Don't have the latest CLI?

curl -LsSf https://hf.co/cli/install.sh | bash

Models citing this paper  1

ItsMaxNorm/MedAgentSim-datasets

Text Generation  •

Updated  Mar 31, 2025

•

2

Datasets citing this paper  1

ItsMaxNorm/MedAgentSim-datasets

Viewer  •

Updated  Mar 25

•

744

•

76

•

1

Spaces citing this paper  0

No Space linking this paper

Cite arxiv.org/abs/2405.07960 in a Space README.md to link it from this page.

Collections including this paper  4

Benchmarks

Collection

147 items

•

Updated  Sep 26, 2025

•

5

DVPS Scientific Watch

Collection

Collection of external scientific material relevant to the project

•

56 items

•

Updated  4 days ago

•

4

Agents

Collection

182 items

•

Updated  Sep 26, 2025

•

3

Agents

Collection

4 items

•

Updated  May 5, 2025

System theme

Company

TOS

Privacy

About

Careers

Website

Models

Datasets

Spaces

Pri
