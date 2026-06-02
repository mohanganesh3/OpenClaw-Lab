[2403.10588] S3LLM: Large-Scale Scientific Software Understanding with LLMs using Source, Metadata, and Document

1

1  institutetext:  University of North Texas, Denton, TX 76207, USA

1

1  email:  kareembabashaik@my.unt.edu,{heng.fan, yunhe.feng}@unt.edu

2

2  institutetext:  Oak Ridge National Laboratory, Oak Ridge, TN 37830, USA

2

2  email:  {wangd,schwartzpd}@ornl.gov

3

3  institutetext:  Argonne National Laboratory, Lemont, IL 60439, USA

3

3  email:  wzheng@anl.gov

4

4  institutetext:  Saint Louis University, St. Louis, MO 63103, USA

4

4  email:  qinglei.cao@slu.edu

S3LLM : Large- S cale  S cientific  S oftware Understanding with  LLMs  using Source, Metadata, and Document

Kareem Shaik

11

Dali Wang

22

Weijian Zheng

33

Qinglei Cao

44

Heng Fan

11

Peter Schwartz

22

Yunhe Feng

11

Abstract

The understanding of large-scale scientific software poses significant challenges due to its diverse codebase, extensive code length, and target computing architectures. The emergence of generative AI, specifically large language models (LLMs), provides novel pathways for understanding such complex scientific codes. This paper presents  S3LLM , an LLM-based framework designed to enable the examination of source code, code metadata, and summarized information in conjunction with textual technical reports in an interactive, conversational manner through a user-friendly interface.  S3LLM  leverages open-source LLaMA-2 models to enhance code analysis through the automatic transformation of natural language queries into domain-specific language (DSL) queries. Specifically, it translates these queries into Feature Query Language (FQL), enabling efficient scanning and parsing of entire code repositories. In addition,  S3LLM  is equipped to handle diverse metadata types, including DOT, SQL, and customized formats. Furthermore,  S3LLM  incorporates retrieval augmented generation (RAG) and LangChain technologies to directly query extensive documents.  S3LLM  demonstrates the potential of using locally deployed open-source LLMs for the rapid understanding of large-scale scientific computing software, eliminating the need for extensive coding expertise, and thereby making the process more efficient and effective.
S3LLM is available at  https://github.com/ResponsibleAILab/s3llm .

Keywords:  Large-Scale Scientific Software, Large Language Models, Research Software Analysis, E3SM Land Model, Retrieval Augmented Generation (RAG), LLM, LLaMA, ChatGPT

1  Introduction

Large-scale scientific computing software is crucial in various scientific fields, undergoing extensive development cycles that lead to the formation of intricate software libraries and ecosystems. This complexity stems from the lengthy development periods, ongoing extensions, and evolving development paradigms, making it imperative to provide users with insights into these computing tools. However, understanding such software is a challenging task due to several factors. First, large-scale scientific software often incorporates multiple programming languages, including older languages such as Fortran and Pascal, which poses a significant challenge for contemporary programmers trying to understand the code. Second, the large volume of scientific software, which may encompass millions of lines of code, presents the obstacle to comprehensively understanding each segment of the code. Lastly, the documentation for these software systems is sometimes less than ideal, often lacking detailed explanations, which further complicates the task of gaining a thorough understanding of the software.

To enhance comprehension of large-scale scientific software, numerous tools have been devised to aid in code analysis and documentation. For instance, Doxygen  [ 2 ]  is capable of generating documentation and performing static code analysis for software source trees. Similarly, Sphinx  [ 8 ]  is compatible with a wide range of programming languages, making it especially effective for producing exhaustive documentation across various formats, including HTML, LaTeX (for printable PDF versions), ePub, Texinfo, manual pages, and plain text. Nonetheless, the currently available tools are primarily tailored for static code analysis and lack the capability to accommodate dynamic queries. Moreover, given the complexity inherent in large-scale scientific software, it poses a significant challenge for both developers and users to formulate queries in both instructed (e.g., textural documents) and structured formats (e.g., SQL). Thus, it is imperative to devise methods for understanding and parsing large-scale scientific software that are both user-friendly and precise.

The emergence of generative AI, particularly large language models (LLMs), heralds a new era in software comprehension and interaction. LLMs have shown remarkable capabilities across various tasks, including chatbot interactions  [ 31 ,  10 ,  15 ] , text summarization  [ 25 ,  12 ,  29 ] , and content creation  [ 9 ,  17 ,  18 ] , demonstrating their potential to revolutionize programming and documentation practices. Beyond these applications, LLMs offer promising solutions for navigating and understanding the complex landscapes of large-scale scientific software  [ 24 ] . By leveraging LLMs, we can envision a future where software comprehension is not only more accessible but also more intuitive, enabling users to query and interact with software in natural language. This paper introduces  S3LLM , a novel framework that embodies this vision, providing a user-friendly interface for interacting with complex scientific computing software through conversational, natural language queries.  S3LLM  aims to bridge the gap between the intricate world of scientific software and the diverse community of users and developers, fostering a deeper understanding and facilitating more effective use of these critical computational tools.

Different from most existing works on software understanding, the proposed  S3LLM  can handle various types of tasks for large-scale scientific software understanding including source code query, metadata analysis, and text-based technical report understanding.  S3LLM  is capable of conducting queries over the information extracted from source code in diverse formats, such as DOT (graph description language)  1

1  1  https://en.wikipedia.org/wiki/DOT_(graph_description_language)

and relational database. By leveraging the few-shot learning capability of LLMs,  S3LLM  can also generate domain-specific language (DSL) queries, such as Feature Query Language (FQL)

[ 33 ] , to gather and extract software features through code analysis. Furthermore,  S3LLM  implements LangChain and Retrieval-Augmented Generation (RAG)  [ 20 ]  schemes to enable text-based queries from technical reports and project summaries. More importantly, all aforementioned interactions and inquiries facilitated by  S3LLM  are executed utilizing natural language.

The contributions of this paper are summarized as follows:

•

We have conceptualized, designed, and implemented  S3LLM , a novel framework that utilizes LLMs to improve the understanding of large-scale scientific software. This framework excels in analyzing source code, metadata, and textual technical reports, providing a holistic approach to software comprehension.

•

S3LLM  presents a user-friendly interface that employs natural language processing, allowing users, even those with limited programming knowledge, to easily query and gain insights into scientific software.

•

Recognizing the need to balance inference speed with the framework’s computational demands,  S3LLM  provides three options featuring LLaMA-2 models with 7B, 13B, and 70B parameters, allowing users to choose the most appropriate model based on their specific requirements.

•

Experiments conducted with the large-scale Energy Exascale Earth System Model (E3SM)  [ 16 ]  demonstrate the effectiveness of our model in analyzing sour