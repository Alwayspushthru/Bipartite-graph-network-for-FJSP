#!/usr/bin/env python3
"""Build a red-marked manuscript draft and a point-by-point response DOCX."""

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import zipfile

from lxml import etree


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "revise/manuscript_changes/manuscript_0.docx"
REVISED = ROOT / "revise/manuscript_changes/manuscript_revised_draft.docx"
RESPONSE = ROOT / "revise/response_materials/response_to_reviewer_draft.docx"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML = "http://www.w3.org/XML/1998/namespace"
NS = {"w": W}


def q(tag):
    return f"{{{W}}}{tag}"


def paragraph_text(p):
    return "".join(p.xpath(".//w:t/text()", namespaces=NS))


def text_run(text, bold=False, italic=False, size=18, color="000000"):
    r = etree.Element(q("r"))
    rpr = etree.SubElement(r, q("rPr"))
    etree.SubElement(rpr, q("color")).set(q("val"), color)
    etree.SubElement(rpr, q("sz")).set(q("val"), str(size))
    etree.SubElement(rpr, q("szCs")).set(q("val"), str(size))
    if bold:
        etree.SubElement(rpr, q("b"))
    if italic:
        etree.SubElement(rpr, q("i"))
    t = etree.SubElement(r, q("t"))
    if text.startswith(" ") or text.endswith(" "):
        t.set(f"{{{XML}}}space", "preserve")
    t.text = text
    return r


def red_run(text, bold=False, italic=False, size=18):
    return text_run(text, bold=bold, italic=italic, size=size, color="FF0000")


def new_paragraph(text, template=None, bold=False, italic=False, size=18, keep_next=False):
    p = etree.Element(q("p"))
    if template is not None:
        ppr = template.find("w:pPr", namespaces=NS)
        if ppr is not None:
            p.append(deepcopy(ppr))
    if keep_next:
        ppr = p.find("w:pPr", namespaces=NS)
        if ppr is None:
            ppr = etree.Element(q("pPr"))
            p.insert(0, ppr)
        if ppr.find("w:keepNext", namespaces=NS) is None:
            etree.SubElement(ppr, q("keepNext"))
    p.append(red_run(text, bold=bold, italic=italic, size=size))
    return p


def response_paragraph(parts, template=None, align=None, keep_next=False):
    p = etree.Element(q("p"))
    if template is not None:
        ppr = template.find("w:pPr", namespaces=NS)
        if ppr is not None:
            p.append(deepcopy(ppr))
    ppr = p.find("w:pPr", namespaces=NS)
    if ppr is None:
        ppr = etree.Element(q("pPr"))
        p.insert(0, ppr)
    if align:
        jc = ppr.find("w:jc", namespaces=NS)
        if jc is None:
            jc = etree.SubElement(ppr, q("jc"))
        jc.set(q("val"), align)
    if keep_next and ppr.find("w:keepNext", namespaces=NS) is None:
        etree.SubElement(ppr, q("keepNext"))
    for text, bold, italic, size in parts:
        p.append(text_run(text, bold=bold, italic=italic, size=size))
    return p


def insert_after(anchor, node):
    parent = anchor.getparent()
    parent.insert(parent.index(anchor) + 1, node)


def set_cell_margins(tc, value=90):
    tcpr = tc.find("w:tcPr", namespaces=NS)
    if tcpr is None:
        tcpr = etree.SubElement(tc, q("tcPr"))
    mar = etree.SubElement(tcpr, q("tcMar"))
    for side in ("top", "left", "bottom", "right"):
        el = etree.SubElement(mar, q(side))
        el.set(q("w"), str(value))
        el.set(q("type"), "dxa")


def make_table(headers, rows, widths, color="FF0000"):
    tbl = etree.Element(q("tbl"))
    tblpr = etree.SubElement(tbl, q("tblPr"))
    etree.SubElement(tblpr, q("tblStyle")).set(q("val"), "TableGrid")
    tblw = etree.SubElement(tblpr, q("tblW"))
    tblw.set(q("w"), str(sum(widths)))
    tblw.set(q("type"), "dxa")
    borders = etree.SubElement(tblpr, q("tblBorders"))
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        b = etree.SubElement(borders, q(edge))
        b.set(q("val"), "single")
        b.set(q("sz"), "4")
        b.set(q("color"), "B7B7B7")
    grid = etree.SubElement(tbl, q("tblGrid"))
    for width in widths:
        etree.SubElement(grid, q("gridCol")).set(q("w"), str(width))

    for ridx, values in enumerate([headers] + rows):
        tr = etree.SubElement(tbl, q("tr"))
        if ridx == 0:
            trpr = etree.SubElement(tr, q("trPr"))
            etree.SubElement(trpr, q("tblHeader"))
        for width, value in zip(widths, values):
            tc = etree.SubElement(tr, q("tc"))
            tcpr = etree.SubElement(tc, q("tcPr"))
            tcw = etree.SubElement(tcpr, q("tcW"))
            tcw.set(q("w"), str(width))
            tcw.set(q("type"), "dxa")
            if ridx == 0:
                shd = etree.SubElement(tcpr, q("shd"))
                shd.set(q("fill"), "EDEDED")
            set_cell_margins(tc)
            p = etree.SubElement(tc, q("p"))
            p.append(text_run(str(value), bold=(ridx == 0), size=16, color=color))
    return tbl


def enable_tracking(settings_root):
    if settings_root.find("w:trackRevisions", namespaces=NS) is None:
        settings_root.insert(0, etree.Element(q("trackRevisions")))


def set_single_column(sectpr, margin=1440):
    for child in list(sectpr):
        if child.tag in (q("headerReference"), q("footerReference"), q("type")):
            sectpr.remove(child)
    pgmar = sectpr.find("w:pgMar", namespaces=NS)
    if pgmar is None:
        pgmar = etree.SubElement(sectpr, q("pgMar"))
    for side in ("top", "right", "bottom", "left"):
        pgmar.set(q(side), str(margin))
    pgmar.set(q("gutter"), "0")
    cols = sectpr.find("w:cols", namespaces=NS)
    if cols is None:
        cols = etree.SubElement(sectpr, q("cols"))
    cols.set(q("num"), "1")
    cols.set(q("space"), "720")


def section_break(sectpr):
    p = etree.Element(q("p"))
    ppr = etree.SubElement(p, q("pPr"))
    ppr.append(deepcopy(sectpr))
    return p


def build_revised_manuscript():
    with zipfile.ZipFile(SOURCE, "r") as zin:
        doc = etree.fromstring(zin.read("word/document.xml"))
        settings = etree.fromstring(zin.read("word/settings.xml"))
        enable_tracking(settings)
        body = doc.find("w:body", namespaces=NS)
        paragraphs = body.findall("w:p", namespaces=NS)

        def find_start(prefix):
            for p in paragraphs:
                if paragraph_text(p).startswith(prefix):
                    return p
            raise RuntimeError(f"Paragraph not found: {prefix}")

        p_features = find_start("The graph is described by operation features")
        p_reward = find_start("where  is the maximum lower-bound completion time")
        p_gru = find_start("Scheduling decisions depend on context accumulated")
        p_conclusion = find_start("This paper presents an end-to-end DRL framework")
        p_references = find_start("6References")
        body_template = p_features
        heading_template = p_references

        feature_clarification = (
            "The retained operation representation also summarizes the unscheduled suffix of each job through "
            "the normalized number of remaining operations and their expected workload. The machine features "
            "describe current availability, expected workload, accumulated idle time, and utilization. These "
            "quantities provide compact progress and load information, but they do not preserve the full "
            "distribution of waiting operations or explicitly include assigned queue length. Appendix A gives "
            "the complete definitions and normalization procedures for all features used by the final model."
        )
        insert_after(p_features, new_paragraph(feature_clarification, body_template))

        reward_clarification = (
            "The reward is evaluated after every scheduling decision and is therefore dense. When gamma = 1, "
            "the undiscounted return telescopes to C_LB(s_0) - C_LB(s_T), which aligns the accumulated stepwise "
            "feedback with the terminal lower-bound objective. This construction provides intermediate feedback "
            "but does not eliminate the credit-assignment difficulty in long scheduling sequences."
        )
        insert_after(p_reward, new_paragraph(reward_clarification, body_template))

        gru_clarification = (
            "The GRUCell contains input-dependent reset and update gates, so the history update is conditioned "
            "on the current graph representation rather than being a fixed linear fusion. The model does not, "
            "however, introduce an additional externally parameterized urgency gate beyond the internal GRU gates."
        )
        insert_after(p_gru, new_paragraph(gru_clarification, body_template))

        limitation = (
            "The compact representation trades state size for information resolution. It retains coarse summaries "
            "of future job workload and machine load, but it does not explicitly encode the full distribution of "
            "waiting operations, assigned machine queues, or a separate urgency signal for memory control. The "
            "dense lower-bound reward also leaves long-horizon credit assignment as an open challenge. Future work "
            "will examine richer state summaries and adaptive learning signals while preserving computational efficiency."
        )
        insert_after(p_conclusion, new_paragraph(limitation, body_template))

        appendix_nodes = []
        appendix_nodes.append(new_paragraph("Appendix A. Detailed Definitions of State Features", heading_template, bold=True, size=20, keep_next=True))
        appendix_nodes.append(new_paragraph(
            "This appendix defines the features used by the final model. Reviewer-motivated diagnostic variants are not part of the final architecture. "
            "For each instance, eligible processing times are first scaled as p_norm = (p - p_min)/(p_max - p_min + epsilon); ineligible entries remain zero. "
            "Operation and machine feature channels are subsequently standardized within each state using their active nodes.",
            body_template,
        ))
        appendix_nodes.append(new_paragraph("A.1 Operation features", heading_template, bold=True, size=18, keep_next=True))
        op_rows = [
            ("x^O_1", "Feasible-machine ratio", "Number of eligible machines for the retained operation divided by the total number of machines."),
            ("x^O_2", "Job-ready time", "Earliest time at which the retained operation can start from the job-precedence perspective."),
            ("x^O_3", "Remaining-operation ratio", "Number of operations from the retained operation to the end of the job, divided by the job length."),
            ("x^O_4", "Remaining expected workload", "Sum of the mean normalized processing times of the retained and subsequent unscheduled operations."),
            ("x^O_5", "Mean processing time", "Mean normalized processing time of the retained operation over its eligible machines."),
            ("x^O_6", "Processing-time span", "Difference between the maximum and minimum normalized processing times over eligible machines."),
            ("x^O_7", "Delay ratio", "log(1 + max(0, t_job - C_LB)/(C_LB + epsilon)) multiplied by one minus the feasible-machine ratio."),
            ("x^O_8", "Criticality", "One minus the lower-bound slack of the job divided by the current global makespan lower bound."),
        ]
        appendix_nodes.append(make_table(("Symbol", "Feature", "Definition"), op_rows, (900, 2100, 6100)))
        appendix_nodes.append(new_paragraph(
            "Completed jobs are masked. For every active feature channel, the mean and population standard deviation are computed over unfinished jobs, and the standardized value is (x - mean)/(standard deviation + epsilon).",
            body_template,
        ))
        appendix_nodes.append(new_paragraph("A.2 Machine features", heading_template, bold=True, size=18, keep_next=True))
        machine_rows = [
            ("x^M_1", "Feasible-operation ratio", "Number of retained candidate operations eligible for the machine divided by the number of unfinished jobs."),
            ("x^M_2", "Machine-ready time", "Completion time of the last operation assigned to the machine."),
            ("x^M_3", "Expected workload", "Sum of eligible processing times over all unscheduled operations, weighted uniformly across each operation's eligible machines."),
            ("x^M_4", "Accumulated idle time", "Idle time accumulated on the machine in the partial schedule."),
            ("x^M_5", "Utilization", "(Machine-ready time - accumulated idle time)/(machine-ready time + epsilon)."),
        ]
        appendix_nodes.append(make_table(("Symbol", "Feature", "Definition"), machine_rows, (900, 2100, 6100)))
        appendix_nodes.append(new_paragraph(
            "Each machine feature channel is standardized across all machines in the current state using (x - mean)/(standard deviation + epsilon).",
            body_template,
        ))
        appendix_nodes.append(new_paragraph("A.3 Operation-machine pair features", heading_template, bold=True, size=18, keep_next=True))
        pair_rows = [
            ("x^P_1", "Processing time", "Normalized processing time of the retained operation on the machine."),
            ("x^P_2", "Earliest start time", "Maximum of the job-ready time and machine-ready time."),
            ("x^P_3", "Job-side delay", "max(0, job-ready time - machine-ready time)."),
            ("x^P_4", "Machine-side delay", "max(0, machine-ready time - job-ready time)."),
            ("x^P_5", "Machine-relative ratio", "Pair processing time divided by the mean processing time of feasible retained operations on the same machine."),
            ("x^P_6", "Operation-relative ratio", "Pair processing time divided by the mean processing time of the same operation over eligible machines."),
        ]
        appendix_nodes.append(make_table(("Symbol", "Feature", "Definition"), pair_rows, (900, 2100, 6100)))
        appendix_nodes.append(new_paragraph(
            "Features for ineligible operation-machine pairs are set to zero and excluded from attention, pooling, and action selection by the dynamic pair mask. epsilon = 10^-8 is used in all denominators for numerical stability.",
            body_template,
        ))

        ref_index = body.index(p_references)
        final_sectpr = body.find("w:sectPr", namespaces=NS)
        preceding_two_column = section_break(final_sectpr)
        appendix_single_column = deepcopy(final_sectpr)
        set_single_column(appendix_single_column)
        appendix_end = section_break(appendix_single_column)
        bounded_appendix = [preceding_two_column] + appendix_nodes + [appendix_end]
        for offset, node in enumerate(bounded_appendix):
            body.insert(ref_index + offset, node)

        overrides = {
            "word/document.xml": etree.tostring(doc, xml_declaration=True, encoding="UTF-8", standalone="yes"),
            "word/settings.xml": etree.tostring(settings, xml_declaration=True, encoding="UTF-8", standalone="yes"),
        }
        with zipfile.ZipFile(REVISED, "w", zipfile.ZIP_DEFLATED) as zout:
            for info in zin.infolist():
                zout.writestr(info.filename, overrides.get(info.filename, zin.read(info.filename)))


def build_response_letter():
    comments = [
        "The current compact bipartite graph only retains one candidate operation for each job. It is recommended to add implicit encoding for suboptimal operations in the waiting queue (such as through gate aggregation mechanism) to avoid decision information loss due to excessive compression, especially in operation intensive Brandimarte instances.",
        "The current 6-dimensional paired features lack an explicit expression of the machine load balancing state. It is recommended to add normalized features of the current queue length and cumulative processing time of the machine, so that the attention mechanism can more accurately evaluate the global impact of operation machine matching.",
        "The current GRU integrates graph level representations in a fixed manner. It is recommended to introduce attention gating mechanism to enable adaptive adjustment of historical state updates based on the urgency of current decisions (such as the urgency of remaining project duration), reducing the interference of irrelevant historical information.",
        "The current reward is based on changes in the lower bound completion time. It is recommended to add sparse milestone rewards (such as reaching a certain threshold for the first time) and multi-objective reward components (such as machine utilization) to alleviate the credit allocation problem of PPO in long sequence scheduling.",
    ]
    responses = [
        (
            "We thank the reviewer for highlighting the possible information loss caused by retaining one schedulable operation per job. We agree that the original manuscript did not explain the retained information with sufficient precision. The current operation representation already includes the normalized number and expected workload of all remaining operations, although it does not explicitly preserve the distribution of the waiting operations. We have clarified this distinction in Section 3.1 and added the complete feature definitions in Appendix A.\n\n"
            "To evaluate the suggested direction, we implemented B0+Q. This variant summarizes the waiting operations using their count ratio, workload ratio, mean flexibility, and mean normalized processing time. A learned gate fuses this summary with the retained-operation embedding. Under the fixed seed-300 protocol, B0+Q changes the Brandimarte mean Gap from 21.70% to 20.88%. This 0.83-percentage-point improvement is below the predefined 1-point retention threshold, while the edata and vdata Gaps increase from 13.44% to 15.32% and from 3.51% to 4.64%, respectively. Because the Brandimarte result is threshold-adjacent, we repeated B0+Q with seeds 301 and 302. The corresponding Brandimarte Gaps are 23.05% and 27.64%, producing a three-seed mean of 23.86%. The seed-300 improvement is therefore not reproducible. We retain the original representation and state this limitation explicitly rather than incorporating an unsupported architectural change. This result concerns the tested aggregation design and does not rule out other waiting-operation encoders."
        ),
        (
            "We agree that queue length and cumulative processing load provide direct indicators of machine congestion. We implemented B0+L by broadcasting two normalized machine quantities to each feasible operation-machine pair: the number of assigned operations and the cumulative assigned processing load. These features enter the existing pair-feature MLP and pair-conditioned attention without changing the action space.\n\n"
            "B0+L increases the Brandimarte mean Gap from 21.70% to 28.42%. It also increases the edata, rdata, and vdata Gaps from 13.44%, 8.80%, and 3.51% to 17.87%, 11.84%, and 10.23%, respectively. The tested features therefore do not satisfy the retention criteria and are not included in the final model. They may partly overlap with the existing machine-ready-time, expected-workload, accumulated-idle-time, and utilization features, although the experiment does not establish redundancy as a mechanism. We have revised Section 3.1 and Appendix A to make the existing load-related information and the boundary of the pair representation explicit."
        ),
        (
            "We thank the reviewer for raising the possibility that irrelevant history may affect sequential decisions. We agree that the original wording did not distinguish the internal GRU gates from an additional explicit urgency gate. The current model uses a GRUCell whose reset and update gates are conditioned on the current graph representation and previous hidden state. It does not use a separate externally parameterized urgency signal. We now state this distinction explicitly in Section 3.3.\n\n"
            "We also implemented B0+G, which applies a graph-conditioned vector gate to the GRU candidate update. At seed 300, B0+G changes the Brandimarte mean Gap from 21.70% to 20.89%, an improvement of 0.82 percentage points, while increasing the edata Gap from 13.44% to 15.40%. We therefore repeated this threshold-adjacent result with seeds 301 and 302. Their Brandimarte Gaps are 23.10% and 33.45%, and the three-seed mean is 25.81%. The apparent improvement at seed 300 is not stable. We consequently retain the original GRU and avoid claiming that it optimally filters historical information. This finding applies only to the tested external gate."
        ),
        (
            "We agree that long-horizon credit assignment remains challenging. We have expanded the explanation of the current reward in Section 3.1. The reward is computed after every scheduling decision and is therefore dense rather than terminal-only. With gamma = 1, its undiscounted return telescopes to C_LB(s_0) - C_LB(s_T), aligning the accumulated stepwise feedback with the terminal lower-bound objective. We also acknowledge that this property does not eliminate long-horizon credit-assignment difficulty.\n\n"
            "To test the suggested use of utilization-related feedback without changing the complete-trajectory objective, we implemented B0+R using endpoint-zero, load-aware potential shaping. B0+R increases the Brandimarte mean Gap from 21.70% to 23.19% and increases the edata, rdata, and vdata Gaps to 19.95%, 12.50%, and 4.93%, respectively. We therefore retain the original dense reward. We have added the reward clarification and the corresponding limitation to the revised manuscript."
        ),
    ]
    excerpts = [
        "The retained operation representation also summarizes the unscheduled suffix of each job through the normalized number of remaining operations and their expected workload. [...] These quantities provide compact progress and load information, but they do not preserve the full distribution of waiting operations or explicitly include assigned queue length. Appendix A gives the complete definitions and normalization procedures for all features used by the final model.",
        "The machine features describe current availability, expected workload, accumulated idle time, and utilization. These quantities provide compact progress and load information, but they do not preserve the full distribution of waiting operations or explicitly include assigned queue length.",
        "The GRUCell contains input-dependent reset and update gates, so the history update is conditioned on the current graph representation rather than being a fixed linear fusion. The model does not, however, introduce an additional externally parameterized urgency gate beyond the internal GRU gates.",
        "The reward is evaluated after every scheduling decision and is therefore dense. When gamma = 1, the undiscounted return telescopes to C_LB(s_0) - C_LB(s_T), which aligns the accumulated stepwise feedback with the terminal lower-bound objective. This construction provides intermediate feedback but does not eliminate the credit-assignment difficulty in long scheduling sequences.",
    ]

    result_rows = [
        ("B0", "21.70", "13.44", "8.80", "3.51", "Retained"),
        ("B0+Q (seed 300)", "20.88", "15.32", "8.36", "4.64", "Not retained"),
        ("B0+L (seed 300)", "28.42", "17.87", "11.84", "10.23", "Not retained"),
        ("B0+G (seed 300)", "20.89", "15.40", "9.75", "3.80", "Not retained"),
        ("B0+R (seed 300)", "23.19", "19.95", "12.50", "4.93", "Not retained"),
    ]
    stability_rows = [
        ("B0+Q", "20.88", "23.05", "27.64", "23.86"),
        ("B0+G", "20.89", "23.10", "33.45", "25.81"),
    ]

    with zipfile.ZipFile(SOURCE, "r") as zin:
        doc = etree.fromstring(zin.read("word/document.xml"))
        body = doc.find("w:body", namespaces=NS)
        original = body.findall("w:p", namespaces=NS)
        body_template = original[31]
        heading_template = original[84]
        sectpr = body.find("w:sectPr", namespaces=NS)
        set_single_column(sectpr)
        for child in list(body):
            if child is not sectpr:
                body.remove(child)

        def add_para(text, bold=False, italic=False, size=20, align=None, keep_next=False):
            body.insert(
                len(body) - (1 if sectpr is not None else 0),
                response_paragraph([(text, bold, italic, size)], body_template, align=align, keep_next=keep_next),
            )

        def add_labelled(label, text, italic_text=False):
            body.insert(
                len(body) - (1 if sectpr is not None else 0),
                response_paragraph(
                    [(label, True, False, 19), (text, False, italic_text, 19)], body_template
                ),
            )

        def add_table(node):
            body.insert(len(body) - (1 if sectpr is not None else 0), node)

        add_para("Response to Reviewer", bold=True, size=32, align="center", keep_next=True)
        add_para("Dear Editor and Reviewer,", size=20)
        add_para(
            "We thank the reviewer for the careful evaluation of our manuscript and for the constructive suggestions concerning state compression, machine-load encoding, recurrent memory, and reward design. We revised the manuscript to clarify the existing feature representation, the internal gating mechanism of the GRU, and the dense lower-bound reward. We also added Appendix A, which provides the complete definitions and normalization procedures for the features used by the final model.",
            size=20,
        )
        add_para(
            "To determine whether the suggested extensions should be incorporated, we implemented four controlled diagnostic variants. All variants were generated from the same B0 code state, trained on SD3 instances with 10 jobs and 5 machines for 1,000 updates, and evaluated on 130 public instances using stochastic beam search with width 10. These experiments were conducted in the same CPU environment for direct comparison. The revision-experiment B0 is a frozen comparison reference for these diagnostic variants and does not replace the results already reported in the main experimental section. None of the variants met the predefined retention criteria, so the final architecture remains unchanged. The diagnostic evidence is reported in this response rather than inserted into the main experimental narrative.",
            size=20,
        )
        add_para("Summary of diagnostic results", bold=True, size=24, keep_next=True)
        add_table(make_table(
            ("Configuration", "Brandimarte", "edata", "rdata", "vdata", "Decision"),
            result_rows,
            (2100, 1300, 1100, 1100, 1100, 1900),
            color="000000",
        ))
        add_para(
            "Values are mean Gaps (%); lower values indicate better performance. The seed-300 paired Brandimarte comparisons do not show a statistically significant improvement after Holm correction.",
            italic=True,
            size=17,
        )
        add_table(make_table(
            ("Configuration", "Seed 300", "Seed 301", "Seed 302", "Three-seed mean"),
            stability_rows,
            (2200, 1500, 1500, 1500, 2300),
            color="000000",
        ))
        add_para(
            "Additional seeds were used for B0+Q and B0+G because their seed-300 Brandimarte effects were close to the predefined one-percentage-point retention threshold.",
            italic=True,
            size=17,
        )
        add_para("Point-by-point response", bold=True, size=24, keep_next=True)
        for idx, (comment, response, excerpt) in enumerate(zip(comments, responses, excerpts), start=1):
            add_para(f"Comment {idx}", bold=True, size=22, keep_next=True)
            add_labelled("Reviewer comment: ", comment)
            response_parts = response.split("\n\n")
            add_labelled("Response: ", response_parts[0])
            for extra in response_parts[1:]:
                add_para(extra, size=19)
            location = "3.1 and Appendix A" if idx in (1, 2) else "3.3 and the Conclusion" if idx == 3 else "3.1 and the Conclusion"
            add_labelled(
                "Changes in the manuscript: ",
                f"Section {location}. Final page and line numbers will be added after layout stabilization.",
            )
            add_para("Revised manuscript text:", bold=True, size=19, keep_next=True)
            add_para(excerpt, italic=True, size=19)
        add_para(
            "We hope that these revisions and controlled diagnostic experiments address the reviewer’s concerns while preserving the coherence of the main experimental evaluation.",
            size=20,
        )
        add_para("Sincerely,", size=20)
        add_para("Xungen Li", size=20)
        add_para("on behalf of all authors", size=20)

        overrides = {
            "word/document.xml": etree.tostring(doc, xml_declaration=True, encoding="UTF-8", standalone="yes")
        }
        with zipfile.ZipFile(RESPONSE, "w", zipfile.ZIP_DEFLATED) as zout:
            for info in zin.infolist():
                zout.writestr(info.filename, overrides.get(info.filename, zin.read(info.filename)))


if __name__ == "__main__":
    build_revised_manuscript()
    build_response_letter()
    print(REVISED)
    print(RESPONSE)
