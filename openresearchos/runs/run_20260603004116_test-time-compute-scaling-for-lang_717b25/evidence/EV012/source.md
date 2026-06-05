# EV012: A Comprehensive Multi-Voltage Design Platform for System-Level Validation of Standard Cell Library

URL: https://www.semanticscholar.org/paper/00163bdd0a5744f44fbd07741f6c853b6801b893
Year: 2021
Source: semantic_scholar
Arxiv: n/a

## Abstract

The quality assurance checks for standard cell libraries are generally limited to validation at individual cell-level. In complex custom ASIC designs such as system-on-chips, which comprise of multiple voltage island and power domain blocks communicating over shared bus interconnects, the operating conditions of standard cells are different from the discrete environment they are tested in. Consequently, any design bugs in the standard cell library that go undetected at cell-level but get captured during their usage in chip development adversely affects the time-to-market while also increasing the resource requirement on library designers to resolve the identified issues. To bridge this gap, we introduce a comprehensive design framework which emulates the usage of standard cells in a multi-voltage system-level environment and facilitates testing of their use-case correctness through functional and low-power simulation checks. Our design approach provides the distinct advantage of 100% coverage of logical standard cells of a given library in a single poweraware gate-level netlist. The design infrastructure is highly automated and scalable to different technology nodes which makes it a powerful platform to build system-level validation flows during the early stages of library development, thereby improving the overall quality of standard cell libraries.
