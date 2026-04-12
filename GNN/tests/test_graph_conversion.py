from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from Real_game_of_life.GNN.graph_conversion import (
    GraphBuildConfig,
    assert_graph_equivalent,
    data_to_graph,
    graph_to_data,
)


def _sample_nodes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_uid": ["seq_a", "seq_a", "seq_a", "seq_a", "seq_a"],
            "frame": [0, 0, 0, 1, 1],
            "spot_id": [10, 11, 12, 20, 21],
            "x": [0.0, 1.0, 5.0, 0.0, 0.0],
            "y": [0.0, 0.0, 0.0, 0.0, 2.0],
            "AREA": [100.0, 110.0, 120.0, 130.0, 140.0],
            "SOLIDITY": [0.95, 0.96, 0.97, 0.98, 0.99],
            "is_boundary_cell": [True, False, True, False, True],
            "target_dx": [1.0, np.nan, np.nan, 0.0, np.nan],
            "target_dy": [0.0, np.nan, np.nan, 2.0, np.nan],
        }
    )


def _roundtrip_config(edge_build_mode: str = "radius") -> GraphBuildConfig:
    return GraphBuildConfig(
        node_id_col="spot_id",
        group_cols=("sequence_uid", "frame"),
        node_feature_columns=("x", "y", "AREA", "SOLIDITY", "is_boundary_cell"),
        edge_feature_columns=("dx", "dy", "distance", "unit_dx", "unit_dy"),
        edge_build_mode=edge_build_mode,
        radius=2.1,
        bidirectional=True,
    )


def test_data_to_graph_builds_spatial_edges_inside_frame_groups() -> None:
    graph = data_to_graph(_sample_nodes(), config=_roundtrip_config())

    assert graph.x.shape == (5, 5)
    assert graph.edge_index.shape == (2, 4)
    assert graph.edge_attr.shape == (4, 5)

    edge_pairs = {
        (graph.nodes.at[source, "spot_id"], graph.nodes.at[target, "spot_id"])
        for source, target in graph.edge_index.T
    }
    assert edge_pairs == {(10, 11), (11, 10), (20, 21), (21, 20)}
    assert (10, 12) not in edge_pairs
    assert (11, 20) not in edge_pairs

    distances = graph.edges.set_index(["source_id", "target_id"])["distance"]
    assert distances.loc[(10, 11)] == 1.0
    assert distances.loc[(20, 21)] == 2.0


def test_data_graph_data_roundtrip_preserves_node_and_edge_tables() -> None:
    nodes = _sample_nodes()
    graph = data_to_graph(nodes, config=_roundtrip_config())
    tables = graph_to_data(graph)

    pd.testing.assert_frame_equal(tables.nodes, nodes.reset_index(drop=True), check_dtype=False)
    pd.testing.assert_frame_equal(tables.edges, graph.edges, check_dtype=False)
    assert tables.node_feature_columns == graph.node_feature_columns
    assert tables.edge_feature_columns == graph.edge_feature_columns


def test_graph_data_graph_roundtrip_preserves_graph_arrays() -> None:
    graph = data_to_graph(_sample_nodes(), config=_roundtrip_config())
    tables = graph_to_data(graph)
    rebuilt = data_to_graph(
        tables.nodes,
        tables.edges,
        config=GraphBuildConfig(
            node_id_col=tables.node_id_col,
            edge_source_col=tables.edge_source_col,
            edge_target_col=tables.edge_target_col,
            node_feature_columns=tables.node_feature_columns,
            edge_feature_columns=tables.edge_feature_columns,
            edge_build_mode="provided",
        ),
    )

    assert_graph_equivalent(graph, rebuilt)


def test_provided_trackmate_style_edges_are_mapped_to_node_indices() -> None:
    nodes = _sample_nodes().loc[:2].copy()
    edges = pd.DataFrame(
        {
            "source": [10, 11],
            "target": [11, 12],
            "edge_speed": [0.5, 0.75],
        }
    )

    graph = data_to_graph(
        nodes,
        edges,
        config=GraphBuildConfig(
            node_id_col="spot_id",
            node_feature_columns=("x", "y", "AREA"),
            edge_feature_columns=("dx", "dy", "distance", "edge_speed"),
            edge_build_mode="provided",
        ),
    )

    assert graph.edge_index.tolist() == [[0, 1], [1, 2]]
    assert graph.edges["source_id"].tolist() == [10, 11]
    assert graph.edges["target_id"].tolist() == [11, 12]
    np.testing.assert_allclose(graph.edge_attr[:, 3], np.array([0.5, 0.75], dtype=np.float32))


def test_graph_data_graph_roundtrip_with_empty_edges() -> None:
    cfg = _roundtrip_config(edge_build_mode="none")
    graph = data_to_graph(_sample_nodes(), config=cfg)

    assert graph.edge_index.shape == (2, 0)
    assert graph.edge_attr.shape == (0, 5)

    tables = graph_to_data(graph)
    rebuilt = data_to_graph(
        tables.nodes,
        tables.edges,
        config=GraphBuildConfig(
            node_id_col=tables.node_id_col,
            edge_source_col=tables.edge_source_col,
            edge_target_col=tables.edge_target_col,
            node_feature_columns=tables.node_feature_columns,
            edge_feature_columns=tables.edge_feature_columns,
            edge_build_mode="provided",
        ),
    )

    assert_graph_equivalent(graph, rebuilt)


def test_cell_graph_exports_to_pyg_data() -> None:
    graph = data_to_graph(_sample_nodes(), config=_roundtrip_config())
    data = graph.to_pyg()

    assert tuple(data.x.shape) == graph.x.shape
    assert tuple(data.edge_index.shape) == graph.edge_index.shape
    assert tuple(data.edge_attr.shape) == graph.edge_attr.shape
    assert data.edge_index.dtype == torch.long
    assert data.x.dtype == torch.float32
    assert data.edge_attr.dtype == torch.float32
    assert data.node_ids == graph.node_ids.tolist()
    assert data.node_feature_columns == list(graph.node_feature_columns)
    assert data.edge_feature_columns == list(graph.edge_feature_columns)
    assert tuple(data.target_delta_pos.shape) == (graph.num_nodes, 2)
    assert data.valid_regression_mask.tolist() == [True, False, False, True, False]
