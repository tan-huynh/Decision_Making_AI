"use client";

import { useEffect, useRef } from "react";
import * as d3 from "d3";
import type { AnalysisResult, DecisionInput } from "@/lib/types";

type Props = {
  decision: DecisionInput;
  result: AnalysisResult | null;
  mode: "map" | "score" | "regret" | "tree";
  refreshToken?: number;
  visibleKinds?: string[];
  onNodeSelect?: (node: { label: string; kind: string; detail: string }) => void;
};

type MapNode = {
  id: string;
  label: string;
  kind: "decision" | "option" | "scenario" | "factor" | "evidence";
  value: number;
  detail: string;
  x?: number;
  y?: number;
  fx?: number | null;
  fy?: number | null;
};

type MapLink = {
  source: string | MapNode;
  target: string | MapNode;
  strength: number;
  label: string;
};

function buildMap(decision: DecisionInput, result: AnalysisResult | null, visibleKinds?: string[]) {
  const nodes: MapNode[] = [
    {
      id: "decision",
      label: "Decision",
      kind: "decision",
      value: 38,
      detail: decision.question
    }
  ];
  const links: MapLink[] = [];
  const optionResults = new Map(result?.option_results.map((item) => [item.id, item]) || []);

  for (const option of decision.options) {
    const optionResult = optionResults.get(option.id);
    const optionId = `option:${option.id}`;
    nodes.push({
      id: optionId,
      label: option.name,
      kind: "option",
      value: Math.max(18, Math.abs(optionResult?.risk_adjusted_score || 35)),
      detail: optionResult
        ? `Score ${optionResult.risk_adjusted_score.toFixed(1)} | EU ${optionResult.expected_utility.toFixed(1)} | regret ${optionResult.expected_regret.toFixed(1)}`
        : `Cost ${option.cost} | reversibility ${option.reversibility}`
    });
    links.push({ source: "decision", target: optionId, strength: 0.8, label: "choice" });

    const costId = `cost:${option.id}`;
    const reverseId = `reverse:${option.id}`;
    nodes.push(
      { id: costId, label: `Cost ${option.cost}`, kind: "factor", value: 12 + option.cost, detail: `Chi phí làm giảm utility của ${option.name}` },
      { id: reverseId, label: `Reverse ${(option.reversibility * 100).toFixed(0)}%`, kind: "factor", value: 20 * option.reversibility + 10, detail: `Khả năng đảo ngược giảm downside risk` }
    );
    links.push(
      { source: costId, target: optionId, strength: 0.35, label: "penalty" },
      { source: reverseId, target: optionId, strength: 0.4, label: "risk control" }
    );

    option.scenarios.forEach((scenario, index) => {
      const scenarioId = `scenario:${option.id}:${index}`;
      const scenarioResult = optionResult?.scenarios[index];
      const impact = Math.abs((scenarioResult?.contribution ?? scenario.probability * scenario.utility) || 1);
      nodes.push({
        id: scenarioId,
        label: scenario.name,
        kind: "scenario",
        value: Math.max(10, Math.min(46, impact)),
        detail: `P ${(scenario.probability * 100).toFixed(0)}% | utility ${scenario.utility} | contribution ${(scenarioResult?.contribution ?? scenario.probability * scenario.utility).toFixed(1)}`
      });
      links.push({ source: optionId, target: scenarioId, strength: Math.max(0.2, scenario.probability), label: `${(scenario.probability * 100).toFixed(0)}%` });
    });
  }

  result?.sensitivity.slice(0, 5).forEach((item, index) => {
    const id = `sensitivity:${index}`;
    nodes.push({
      id,
      label: item.variable,
      kind: "evidence",
      value: Math.max(12, Math.min(34, item.impact)),
      detail: `Sensitivity impact ${item.impact.toFixed(1)} on ${item.option}`
    });
    const target = decision.options.find((option) => option.name === item.option);
    if (target) links.push({ source: id, target: `option:${target.id}`, strength: 0.28, label: "impact" });
  });

  if (result) {
    nodes.push({
      id: "voi",
      label: `VOI ${result.value_of_information.score.toFixed(0)}`,
      kind: "evidence",
      value: Math.max(14, Math.min(42, result.value_of_information.score / 2)),
      detail: `Value of information: ${result.value_of_information.most_valuable_unknowns.join("; ")}`
    });
    links.push({ source: "decision", target: "voi", strength: 0.3, label: "unknowns" });
  }

  if (!visibleKinds?.length) return { nodes, links };
  const alwaysVisible = new Set(["decision", "option"]);
  const visible = new Set(visibleKinds);
  const filteredNodes = nodes.filter((node) => alwaysVisible.has(node.kind) || visible.has(node.kind));
  const filteredIds = new Set(filteredNodes.map((node) => node.id));
  return {
    nodes: filteredNodes,
    links: links.filter((link) => {
      const source = typeof link.source === "string" ? link.source : link.source.id;
      const target = typeof link.target === "string" ? link.target : link.target.id;
      return filteredIds.has(source) && filteredIds.has(target);
    }),
  };
}

export default function DecisionChart({ decision, result, mode, refreshToken = 0, visibleKinds, onNodeSelect }: Props) {
  const ref = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    const svgElement = ref.current;
    if (!svgElement) return;
    const svg = d3.select(svgElement);
    svg.selectAll("*").remove();

    const width = svgElement.clientWidth || 720;
    const height = svgElement.clientHeight || 360;
    svg.attr("viewBox", `0 0 ${width} ${height}`);

    if (mode === "map") {
      const { nodes, links } = buildMap(decision, result, visibleKinds);
      const color = d3
        .scaleOrdinal<string, string>()
        .domain(["decision", "option", "scenario", "factor", "evidence"])
        .range(["#202124", "#0f766e", "#2563eb", "#b45309", "#7c3aed"]);
      const g = svg.append("g");
      svg.call(
        d3
          .zoom<SVGSVGElement, unknown>()
          .scaleExtent([0.45, 2.8])
          .on("zoom", (event) => g.attr("transform", event.transform))
      );

      const tooltip = d3
        .select(svgElement.parentElement)
        .append("div")
        .attr("class", "map-tooltip")
        .style("position", "absolute")
        .style("pointer-events", "none")
        .style("opacity", "0");

      const simulation = d3
        .forceSimulation<MapNode>(nodes)
        .force(
          "link",
          d3
            .forceLink<MapNode, MapLink>(links)
            .id((node) => node.id)
            .distance((link) => 80 + 80 * (1 - link.strength))
            .strength((link) => link.strength)
        )
        .force("charge", d3.forceManyBody().strength(-420))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("collision", d3.forceCollide<MapNode>().radius((node) => Math.sqrt(node.value) + 34));

      const link = g
        .append("g")
        .attr("stroke", "#c9c3b8")
        .attr("stroke-opacity", 0.75)
        .selectAll("line")
        .data(links)
        .join("line")
        .attr("stroke-width", (d) => Math.max(1, d.strength * 4));

      const label = g
        .append("g")
        .selectAll("text")
        .data(links)
        .join("text")
        .attr("font-size", 10)
        .attr("fill", "#687076")
        .attr("text-anchor", "middle")
        .text((d) => d.label);

      const node = g
        .append("g")
        .selectAll<SVGGElement, MapNode>("g")
        .data(nodes)
        .join("g")
        .style("cursor", "grab")
        .call(
          d3
            .drag<SVGGElement, MapNode>()
            .on("start", (event, d) => {
              if (!event.active) simulation.alphaTarget(0.25).restart();
              d.fx = d.x;
              d.fy = d.y;
            })
            .on("drag", (event, d) => {
              d.fx = event.x;
              d.fy = event.y;
            })
            .on("end", (event, d) => {
              if (!event.active) simulation.alphaTarget(0);
              d.fx = null;
              d.fy = null;
            })
        )
        .on("mouseenter", (event, d) => {
          tooltip
            .style("opacity", "1")
            .style("left", `${event.offsetX + 14}px`)
            .style("top", `${event.offsetY + 14}px`)
            .html(`<strong>${d.label}</strong><br/>${d.detail}`);
        })
        .on("mousemove", (event) => {
          tooltip.style("left", `${event.offsetX + 14}px`).style("top", `${event.offsetY + 14}px`);
        })
        .on("mouseleave", () => tooltip.style("opacity", "0"));
      node.on("click", (_, d) => onNodeSelect?.({ label: d.label, kind: d.kind, detail: d.detail }));

      node
        .append("circle")
        .attr("r", (d) => Math.sqrt(d.value) + 9)
        .attr("fill", (d) => color(d.kind))
        .attr("fill-opacity", 0.9)
        .attr("stroke", "#fff")
        .attr("stroke-width", 2);

      node
        .append("text")
        .attr("text-anchor", "middle")
        .attr("dy", 4)
        .attr("font-size", 10)
        .attr("font-weight", 700)
        .attr("fill", "#fff")
        .text((d) => (d.label.length > 16 ? `${d.label.slice(0, 16)}...` : d.label));

      simulation.on("tick", () => {
        link
          .attr("x1", (d) => (d.source as MapNode).x || 0)
          .attr("y1", (d) => (d.source as MapNode).y || 0)
          .attr("x2", (d) => (d.target as MapNode).x || 0)
          .attr("y2", (d) => (d.target as MapNode).y || 0);

        label
          .attr("x", (d) => (((d.source as MapNode).x || 0) + ((d.target as MapNode).x || 0)) / 2)
          .attr("y", (d) => (((d.source as MapNode).y || 0) + ((d.target as MapNode).y || 0)) / 2);

        node.attr("transform", (d) => `translate(${d.x || 0},${d.y || 0})`);
      });

      return () => {
        simulation.stop();
        tooltip.remove();
      };
    }

    if (!result) {
      svg
        .append("text")
        .attr("x", width / 2)
        .attr("y", height / 2)
        .attr("text-anchor", "middle")
        .attr("fill", "#687076")
        .text("Chạy phân tích để xem decision map");
      return;
    }

    if (mode === "tree") {
      const root = d3.hierarchy({
        name: "Decision",
        children: result.option_results.map((option) => ({
          name: option.name,
          score: option.risk_adjusted_score,
          children: option.scenarios.map((scenario) => ({
            name: scenario.name,
            score: scenario.contribution,
            probability: scenario.probability
          }))
        }))
      });

      const tree = d3.tree<typeof root.data>().size([height - 50, width - 170]);
      tree(root);

      const g = svg.append("g").attr("transform", "translate(92,25)");
      g.selectAll("path")
        .data(root.links())
        .join("path")
        .attr("fill", "none")
        .attr("stroke", "#c8c2b6")
        .attr("d", (d) => {
          const source = d.source as d3.HierarchyPointNode<typeof root.data>;
          const target = d.target as d3.HierarchyPointNode<typeof root.data>;
          const mid = (source.y + target.y) / 2;
          return `M${source.y},${source.x}C${mid},${source.x} ${mid},${target.x} ${target.y},${target.x}`;
        });

      const node = g
        .selectAll("g")
        .data(root.descendants())
        .join("g")
        .attr("transform", (d) => `translate(${d.y},${d.x})`);

      node
        .append("circle")
        .attr("r", (d) => (d.depth === 0 ? 8 : 6))
        .attr("fill", (d) => (d.depth === 1 ? "#0f766e" : d.depth === 2 ? "#2563eb" : "#202124"));

      node
        .append("text")
        .attr("dy", -10)
        .attr("text-anchor", "middle")
        .attr("font-size", 11)
        .attr("fill", "#202124")
        .text((d) => {
          const name = d.data.name.length > 22 ? `${d.data.name.slice(0, 22)}...` : d.data.name;
          return d.depth === 2 && "probability" in d.data ? `${name} ${(Number(d.data.probability) * 100).toFixed(0)}%` : name;
        });
      return;
    }

    const data = result.option_results.map((d) => ({
      name: d.name,
      value: mode === "score" ? d.risk_adjusted_score : d.expected_regret
    }));
    const margin = { top: 22, right: 24, bottom: 70, left: 58 };
    const x = d3
      .scaleBand()
      .domain(data.map((d) => d.name))
      .range([margin.left, width - margin.right])
      .padding(0.25);
    const y = d3
      .scaleLinear()
      .domain([0, Math.max(10, d3.max(data, (d) => d.value) || 0)])
      .nice()
      .range([height - margin.bottom, margin.top]);

    svg
      .append("g")
      .attr("transform", `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(x))
      .selectAll("text")
      .attr("transform", "rotate(-24)")
      .attr("text-anchor", "end")
      .attr("font-size", 11);

    svg.append("g").attr("transform", `translate(${margin.left},0)`).call(d3.axisLeft(y).ticks(6));

    svg
      .selectAll("rect")
      .data(data)
      .join("rect")
      .attr("x", (d) => x(d.name) || 0)
      .attr("y", (d) => y(d.value))
      .attr("width", x.bandwidth())
      .attr("height", (d) => height - margin.bottom - y(d.value))
      .attr("rx", 4)
      .attr("fill", mode === "score" ? "#0f766e" : "#b45309");

    svg
      .selectAll(".bar-label")
      .data(data)
      .join("text")
      .attr("class", "bar-label")
      .attr("x", (d) => (x(d.name) || 0) + x.bandwidth() / 2)
      .attr("y", (d) => y(d.value) - 8)
      .attr("text-anchor", "middle")
      .attr("font-size", 12)
      .attr("font-weight", 700)
      .text((d) => d.value.toFixed(1));
  }, [decision, result, mode, refreshToken, visibleKinds, onNodeSelect]);

  return <svg ref={ref} className="chart" role="img" aria-label="Decision analysis chart" />;
}
