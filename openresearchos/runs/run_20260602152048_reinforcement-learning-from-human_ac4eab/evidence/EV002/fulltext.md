[2409.06957] Formatting Instructions for ICLR 2025 Conference Submissions

Formatting Instructions for ICLR 2025

Conference Submissions

Antiquus S. Hippocampus, Natalia Cerebro &amp; Amelie P. Amygdale

Department of Computer Science
 Cranberry-Lemon University
 Pittsburgh, PA 15213, USA

{hippo,brain,jen}@cs.cranberry-lemon.edu

\And Ji Q. Ren &amp; Yevgeny LeNet

Department of Computational Neuroscience

University of the Witwatersrand

Joburg, South Africa

{robot,net}@wits.ac.za

\AND Coauthor

Affiliation

Address

email

Use footnote for providing further information
about author (webpage, alternative address)— not  for acknowledging
funding agencies. Funding acknowledgements go at the end of the paper.

Abstract

The abstract paragraph should be indented 1/2 inch (3 picas) on both left and
right-hand margins. Use 10 point type, with a vertical spacing of 11 points.
The word  Abstract  must be centered, in small caps, and in point size 12. Two
line spaces precede the abstract. The abstract must be limited to one
paragraph.

1  Submission of conference papers to ICLR 2025

ICLR requires electronic submissions, processed by
 https://openreview.net/ . See ICLR’s website for more instructions.

If your paper is ultimately accepted, the statement  \iclrfinalcopy  should be inserted to adjust the
format to the camera ready requirements.

The format for the submissions is a variant of the NeurIPS format.
Please read carefully the instructions below, and follow them
faithfully.

1.1  Style

Papers to be submitted to ICLR 2025 must be prepared according to the
instructions presented here.

Authors are required to use the ICLR  L a T e X  style files obtainable at the
ICLR website. Please make sure you use the current files and
not previous versions. Tweaking the style files may be grounds for rejection.

1.2  Retrieval of style files

The style files for ICLR and other conference information are available online at:

http://www.iclr.cc/

The file  iclr2025_conference.pdf  contains these
instructions and illustrates the
various formatting requirements your ICLR paper must satisfy.
Submissions must be made using  L a T e X  and the style files
 iclr2025_conference.sty  and  iclr2025_conference.bst  (to be used with  L a T e X 2e). The file
 iclr2025_conference.tex  may be used as a “shell” for writing your paper. All you
have to do is replace the author, title, abstract, and text of the paper with
your own.

The formatting instructions contained in these style files are summarized in
sections

2  ,

3  , and

4

below.

2  General formatting instructions

The text must be confined within a rectangle 5.5 inches (33 picas) wide and
9 inches (54 picas) long. The left margin is 1.5 inch (9 picas).
Use 10 point type with a vertical spacing of 11 points. Times New Roman is the
preferred typeface throughout. Paragraphs are separated by 1/2 line space,
with no indentation.

Paper title is 17 point, in small caps and left-aligned.
All pages should start at 1 inch (6 picas) from the top of the page.

Authors’ names are
set in boldface, and each name is placed above its corresponding
address. The lead author’s name is to be listed first, and
the co-authors’ names are set to follow. Authors sharing the
same address can be on the same line.

Please pay special attention to the instructions in section

4

regarding figures, tables, acknowledgments, and references.

There will be a strict upper limit of 10 pages for the main text of the initial submission, with unlimited additional pages for citations.

3  Headings: first level

First level headings are in small caps,
flush left and in point size 12. One line space before the first level
heading and 1/2 line space after the first level heading.

3.1  Headings: second level

Second level headings are in small caps,
flush left and in point size 10. One line space before the second level
heading and 1/2 line space after the second level heading.

3.1.1  Headings: third level

Third level headings are in small caps,
flush left and in point size 10. One line space before the third level
heading and 1/2 line space after the third level heading.

4  Citations, figures, tables, references

These instructions apply to everyone, regardless of the formatter being used.

4.1  Citations within the text

Citations within the text should be based on the  natbib  package
and include the authors’ last names and year (with the “et al.” construct
for more than two authors). When the authors or the publication are
included in the sentence, the citation should not be in parenthesis using  \citet{}  (as
in “See

Hinton06

for more information.”). Otherwise, the citation
should be in parenthesis using  \citep{}  (as in “Deep learning shows promise to make progress
towards AI  ( Bengio+chapter2007 ) .”).

The corresponding references are to be listed in alphabetical order of
authors, in the  References  section. As to the format of the
references themselves, any style is acceptable as long as it is used
consistently.

4.2  Footnotes

Indicate footnotes with a number  1

1  1 Sample of the first footnote

in the
text. Place the footnotes at the bottom of the page on which they appear.
Precede the footnote with a horizontal rule of 2 inches
(12 picas).  2

2  2 Sample of the second footnote

4.3  Figures

All artwork must be neat, clean, and legible. Lines should be dark
enough for purposes of reproduction; art work should not be
hand-drawn. The figure number and caption always appear after the
figure. Place one line space before the figure caption, and one line
space after the figure. The figure caption is lower case (except for
first word and proper nouns); figures are numbered consecutively.

Make sure the figure caption does not get separated from the figure.
Leave sufficient space to avoid splitting the figure and figure caption.

You may use color figures.
However, it is best for the
figure captions and the paper body to make sense if the paper is printed
either in black/white or in color.

Figure 1:  Sample figure caption.

4.4  Tables

All tables must be centered, neat, clean and legible. Do not use hand-drawn
tables. The table number and title always appear before the table. See
Table

1  .

Place one line space before the table title, one line space after the table
title, and one line space after the table. The table title must be lower case
(except for first word and proper nouns); tables are numbered consecutively.

Table 1:  Sample table title

PART

DESCRIPTION

Dendrite

Input terminal

Axon

Output terminal

Soma

Cell body (contains cell nucleus)

5  Default Notation

In an attempt to encourage standardized notation, we have included the
notation file from the textbook,  Deep Learning

goodfellow2016deep

available at
 https://github.com/goodfeli/dlbook_notation/ . Use of this style
is not required and can be disabled by commenting out
 math_commands.tex .

Numbers and Arrays

a

𝑎

\displaystyle a

A scalar (integer or real)

\va

\va

\displaystyle\va

A vector

\mA

\mA

\displaystyle\mA

A matrix

\tA

\tA

\displaystyle\tA

A tensor

\mI

n

subscript

\mI

𝑛

\displaystyle\mI_{n}

Identity matrix with

n

𝑛

n

rows and

n

𝑛

n

columns

\mI

\mI

\displaystyle\mI

Identity matrix with dimensionality implied by context

\ve

(  i  )

superscript

\ve

𝑖

\displaystyle\ve^{(i)}

Standard basis vector

[  0  ,  …  ,  0  ,  1  ,  0  ,  …  ,  0  ]

0  …  0  1  0  …  0

[0,\dots,0,1,0,\dots,0]

with a 1 at position

i

𝑖

i

\text

​  d  ​  i  ​  a  ​  g  ​

(

\va

)

\text

𝑑  𝑖  𝑎  𝑔

\va

\displaystyle\text{diag}(\va)

A square, diagonal matrix with diagonal entries given by

\va

\va

\va

\ra

\ra

\displaystyle\ra

A scalar random variable

\rva

\rva

\displaystyle\rva

A vector-valued random variable

\rmA

\rmA

\displaystyle\rmA

A matrix-valued random variable

Sets and Graphs

\sA

\sA

\displaystyle\sA

A set

\R

\R

\displaystyle\R

The set of real numbers

{  0  ,  1  }

0  1

\displaystyle\{0,