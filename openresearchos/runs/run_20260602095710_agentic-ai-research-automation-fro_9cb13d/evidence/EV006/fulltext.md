[2601.06064] Socio-technical aspects of Agentic AI

Socio-technical aspects of Agentic AI

Praveen Kumar Donta

praveen@dsv.su.se

0000-0002-8233-6071

Department of Computer and Systems Sciences, Stockholm University  Stockholm  Sweden  106 91

,

Alaa Saleh

0009-0009-6317-2823

alaa.saleh@oulu.fi

Center for Ubiquitous Computing, University of Oulu  Oulu  Finland  90014

,

Ying Li

liying1771@163.com

0000-0002-6585-0714

College of Computer Science and Engineering, Northeastern University  Shenyang  China

,

Shubham Vaishnav

shubham.vaishnav@dsv.su.se

0000-0001-7612-4227

Department of Computer and Systems Sciences, Stockholm University  Stockholm  Sweden  106 91

,

Kai Fang

Kaifang@zafu.edu.cn

,

Hailin Feng

hlfeng@zafu.edu.cn

0000-0002-8233-6071

,

Yuchao Xia

xyc@stu.zafu.edu.cn

,

Thippa Reddy Gadekallu

thippa@zhongda.cn

0000-0003-0097-801X

Zhejiang A&amp;F University, Hangzhou  Hangzhou  China  311300

,

Qiyang Zhang

0000-0001-5585-6613

qiyangzhang@pku.edu.cn

School of Computer Science, Peking University  Beijing  100087  China

,

Xiaodan Shi

xiaodan.shi@dsv.su.se

0000-0001-5125-1860

,

Ali Beikmohammadi

beikmohammadi@dsv.su.se

0000-0003-4884-4600

,

Sindri Magnússon

sindri.magnusson@dsv.su.se

0000-0002-6617-8683

Department of Computer and Systems Sciences, Stockholm University  Stockholm  Sweden  106 91

,

Ilir Murturi

0000-0003-0240-3834

ilir.murturi@uni-pr.edu

Department of Mechatronics, University of Prishtina  Prishtina  100087  Kosova

,

Chinmaya Kumar Dehury

0000-0003-1990-0431

dehury@iiserbpr.ac.in

Department of Computer Science, IISER Berhampur  Berhampur, Odisha  India  760010

,

Marcin Paprzycki

marcin.paprzycki@ibspan.waw.pl

0000-0002-8069-2152

Systems Research Institute Polish Academy of Sciences  Warsaw  Poland  01-447

,

Lauri Loven

0000-0001-9475-4839

lauri.loven@oulu.fi

Center for Ubiquitous Computing, University of Oulu  Oulu  Finland  90014

,

Sasu Tarkoma

0000-0003-4220-3650

sasu.tarkoma@helsinki.fi

Department of Computer Science, University of Helsinki  Helsinki  Finland  00100

and

Schahram Dustdar

0000-0001-6872-8821

schahram.dustdar@upf.edu

ICREA, Barcelona  Barcelona  Spain  08002

(2026)

Abstract.

Agentic Artificial Intelligence (AI) represents a fundamental shift in the design of intelligent systems, characterized by interconnected components that collectively enable autonomous perception, reasoning, planning, action, and learning. Recent research on agentic AI has largely focused on technical foundations, including system architectures, reasoning and planning mechanisms, coordination strategies, and application-level performance across domains. However, the societal, ethical, economic, environmental, and governance implications of agentic AI remain weakly integrated into these technical treatments. This paper addresses this gap by presenting a socio-technical analysis of agentic AI that explicitly connects core technical components with societal context. We examine how architectural choices in perception, cognition, planning, execution, and memory introduce dependencies related to data governance, accountability, transparency, safety, and sustainability. To structure this analysis, we adopt the MAD–BAD–SAD construct as an analytical lens, capturing motivations, applications, and moral dilemmas (MAD); biases, accountability, and dangers (BAD); and societal impact, adoption, and design considerations (SAD). Using this lens, we analyze ethical considerations, implications, and challenges arising from contemporary agentic AI systems and assess their manifestation across emerging applications, including healthcare, education, industry, smart and sustainable cities, social services, communications and networking, and earth observation and satellite communications. The paper further identifies open challenges and suggests future research directions, framing agentic AI as an integrated socio-technical system whose behavior and impact are co-produced by algorithms, data, organizational practices, regulatory frameworks, and social norms.

Agentic AI; Societal and Technical AI; AI Ethics;

†

†  copyright:  acmlicensed

†

†  journalyear:  2026

†

†  doi:  XXXXXXX.XXXXXXX

†

†  journal:  CSUR

1.  Introduction

Nowadays, it is no exaggeration to say AI is becoming ubiquitous in everyday life  (Denning,  2025 ) . For example, from large-scale applications like space exploration and Earth Observation (EO)  (Ruan  et al. ,  2025a )  to smaller-scale uses such as smart home devices and personal health trackers, AI continues to become ubiquitous, being essential to various domains and daily life innovations. There is no surprise in saying that these innovations are advancing through AI, making it likely that almost every application and object will incorporate some level of AI significance in the near future. As technology evolves, agentic AI is becoming the next vital stage in AI. Unlike traditional AI agents which perform specific tasks based on set instructions, agentic AI operates autonomously, independently planning and making decisions to achieve complex goals  (Murugesan,  2025 ; Huang,  2025 ; Hughes and others,  2025 ) . It learns, adapts, and carries out multi-step actions with minimal or no human input. These capabilities offer agentic AI distinct advantages over traditional AI agents, especially in managing dynamic and complex environments that demand continuous adaptability and goal-oriented behavior  (Sapkota  et al. ,  2026 ) . However, since most real-time applications are inherently goal-oriented, the adoption of agentic AI is rapidly increasing across contemporary domains due to its ability to significantly enhance operational efficiency, scalability, and automation across business, research, and technology space  (Gridach  et al. ,  2025 ) .

As AI continues to advance, agentic AI stands out as a powerful tool that can drive innovation and productivity by operating independently and intelligently in these scenarios  (Hornyak,  2025 ; Denning,  2025 ) . Recent works highlight that agentic AI systems can reason over extended time horizons, coordinate sequences of actions, and adapt their behavior in response to changing environmental conditions. Achieving this autonomy remains challenging for traditional AI systems  (Huang,  2025 ; Murugesan,  2025 ) . Further, these capabilities allow agentic systems to operate effectively in open-ended and dynamic settings, where predefined workflows and static decision rules are insufficient. Increased autonomy introduces fundamental challenges involving reliability under uncertainty, safety in autonomous decision-making, controllability, and alignment with intended objectives. Such challenges are amplified when agentic AI systems operate in open-ended environments and engage in interactions with other agents. Existing literature has primarily addressed these challenges through technical approaches, focusing on agent architectures, planning and learning mechanisms, and performance optimization  (Saleh  et al. ,  2025b ; Deng and others,  2025 ; Yao  et al. ,  2023 ; Wang  et al. ,  2024b ) . Broader societal, ethical, and governance considerations remain inadequately integrated into existing technical frameworks.

Technical advances alone are insufficient to fully account for the behavior and impact of agentic AI systems deployed in real-world settings  (Kapoor  et al. ,  2024 ) . When autonomous agents operate with reduced human oversight and interact directly with social, organizational, and institutional environments, their decisions can shape human behavior, redistribute authority, and produce effects at scale  (Bommasani and others,  2023 ) . These dynamics introduce concerns related to accountability, transparency, trust, and value alignment that extend beyond what can be addressed through architectural or algorithmic solutions alone  (Raji  et al. ,  2020 ) . In practice, the behavior and impact of agentic AI s