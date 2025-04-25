
#### docker & k8s

##### k8s的优缺点？

优点：
- 确保一致性：避免不同环境之间的不一致问题。
- 提高可复现性：保障测试与生产环境一致。
- 减少环境漂移：所有变更均通过新版本管理，防止手动修改配置导致不稳定。
- 方便版本回滚：可以快速切换到稳定版本，提高系统可靠性。

缺点：
- 存储消耗大：不同版本的镜像需要消耗额外的空间。
- 无法直接修改运行中的状态：所有的变更必须通过构建新镜像并重新部署。

##### 如何优化镜像存储，避免浪费？

- docker镜像分层技术：避免重复存储相同的部分，只存储变更内容。
- 使用CI/CD结合镜像清理策略：通过Harbor、docker registry设置镜像生命周期策略，定期删除过期镜像；使用docker image prune命令清理未使用的镜像层。

##### 容器本地配置修改后，重启是否失效？如何避免？

修改后会丢失，因为容器是无状态的。解决方案：
- 使用ConfigMap 和Secret存储配置，适用于环境变量、非敏感和敏感配置信息。
- 将配置文件挂载到Persistent Volume（PV），确保Pod 重启后仍能访问原数据。
- 直接打包到新镜像，适用于固定不变的配置。

##### k8s如何从当前状态变更到目标状态？

kubernetes采用声明式配置（declarative configration）。用户通过YAML文件定义目标状态。控制器（controller）持续监控实际状态，并调整使其与目标状态保持一致。

##### 如何精细控制状态变更？

Rolling Update（滚动升级）：逐步替换Pod ，确保无缝更新，避免服务中断。探针机制（Liveness Probe, Readiness Probe）：Liveness Probe检查容器是否存活，失败则重启；Readiness Probe检查服务是否准备就绪，未就绪的Pod不会接受流量。

#### 容器的基础

镜像：由Dockerfile构建，包含应用代码、依赖和运行环境。容器：由镜像启动的实例，隔离运行应用

##### 容器与镜像的关系，容器与运行时（CRI）是什么？

容器时镜像的启动实例，CRI允许kubernetes与不同容器运行时（Docker、Containerd、CRI-O）通信。

##### Pod为什么通常只运行一个容器？

一个Pod可以运行多个容器，但通常只有一个主容器，其他容器（sidecar）用于收集日志、代理等辅助功能。

##### 升级一个容器的镜像会导致其他容器重启吗？

不会，Pod级别的变更不会影响其他Pod内的容器，除非它们共享相同的网络和存储。

##### 节点池与k8s集群之间的关系？

节点池是一组具有相同配置的节点，一个k8s可以包含多个节点池。

##### ReplicaSet和Deployment的区别？

ReplicaSet只负责维持Pod的副本数，Deployment在ReplicaSet之上，提供滚动升级、回滚等能力

##### 为什么生产环境建议直接使用Deployment而不是ReplicaSet？

Deployment管理ReplicaSet，支持平滑升级、回滚，而单独使用ReplicaSet不具备这些功能。

##### 如何手动向ReplicaSet添加容器？

不推荐直接修改ReplicaSet，应该修改Deployment定义，并让其自动调整ReplicaSet。

##### 升级已有应用如何控制更新顺序？

滚动升级（默认）：逐步更新Pod，确保服务不中断。蓝绿部署：同时运行新旧版本，切换流量。金丝雀发布：先升级部分Pod，观察一段时间后再全面更新。

##### ConfigMap与本地配置文件的关系？

ConfigMap提供动态配置管理，比本地配置文件更加灵活，适用于k8s原生应用。

##### k8s如何实现服务发现？

通过service资源，提供负载均衡和内部DNS解析：ClusterIP（默认，集群内部访问）、NodePort（暴露节点端口）、LoadBalancer（云负载均衡）、ExternalName（外部服务映射）。

#### 容器技术&核心原理

容器是一种基于操作系统内核特性的进程隔离运行环境，其核心思想是实现应用程序级别的虚拟化。容器有如下特性：
- 轻量级虚拟化：容器共享宿主机内核，无需安装操作系统，资源占用少。
- 环境一致性：容器内应用程序的依赖都打包在一起，实现跨平台一致运行。
- 快速启动与部署：容器启动速度远高于传统虚拟机，秒级启动，灵活性高。
- 易于迁移和部署：便于跨平台部署与迁移，实现开发与运维的高效协作。

容器设计思想：资源隔离与封装，不同应用或进程之间互相不干扰，拥有各自独立的运行环境；一次构建，处处运行，确保在任何环境运行结果都相同。不可变基础设施，避免环境漂移，提升系统稳定性。敏捷部署与弹性扩展：快速部署、启动、终止容器，灵活地动态扩缩容。

容器的核心技术：
- 进程隔离（namespaces）：Linux内核中的namespace技术实现了进程级别的资源隔离，常见的namespace类型包括：pid namespace：进程id隔离，容器进程间互不可见。Network namespace：提供独立的网络栈、虚拟网络接口。mount namespace：文件系统挂载点隔离，容器拥有独立的文件系统。ipc namespace：进程间通信（IPC）隔离。UTS namespace：主机名、域名隔离。User namespace：用户和用户组隔离。namespace保证容器内部进程与外部环境互不影响，创造了独立运行空间。
- 资源配额与控制：Cgroups是Linux内核的一种特性，用于实现容器资源管理，包括：限制进程对CPU、内存、磁盘I/O和网络带宽等资源的使用。确保容器的之间的资源隔离，避免资源竞争导致系统性能下降。提高系统稳定性和资源利用效率，实现资源的公平分配。例如，CPU配额控制公式：CPU_Quota = CPU_shares x （容器权重 / 所用容器权重之和），通过合理分配配额，实现资源利用的公平性和高效性。
- 容器进程管理工具：容器的进程管理工具用于管理容器的生命周期，包括容器的创建、运行、监控和终止，常见工具包括：docker engine：提供基础的容器创建、运行和销毁功能。containerd：符合OCI标准的运行时，轻量、高效、kubernetes默认支持的容器运行时。CRI-O转为kubernetes设计的容器运行时，精简高效，易于kubernetes集成。runC：Docker和CRI-O等工具使用的底层运行时，负责实际进程创建于管理。

容器接口标准（OCI和CRI）：容器接口标准化对容器发展至关重要，目前业界最重要的两个标准分别为：OCI（Open Container Initative）和CRI（Container Runtime Interface）。OCI是由Docker与其他容器厂商共同制定的开源容器技术标准，旨在定义容器镜像和运行时环境的统一规范。OCI标准主要有两大规范：
- OCI Runtime Specification（运行时规范）：描述运行时必须提供的标准接口与环境。规范了容器的生命周期（如创建、启动、停止、删除等）。定义了容器的状态和操作接口（如创建、启动、停止、删除容器的API标准）。
- OCI Image Specification（镜像规范）：定义了容器镜像的标准结构。镜像的元数据（Mainfest、配置文件）以及层（Layers）的定义。镜像存储、分发和管理的统一规范。

OCI兼容工具：目前主流的OCI兼容工具包括：Containerd(Docker运行时底层引擎)、CRI-O(专为kubernates优化的容器运行时)、Podman(无守护进程的容器引擎)。

CRI（容器运行时接口）是kubernetes提出的标准接口规范，旨在抽象kubernates与底层容器运行时之间的交互方式，使kubernates支持不同容器运行时环境，CRI规范设计的目的：kubernates与容器运行时之间的标准化交互。隔离kubernates与特定容器运行时的依赖。允许kubernates支持多种容器运行时（如Docker、 Containerd、CRI-O）。

CRI的架构与工作原理：kubernates调用标准的CRI接口操作容器的生命周期，CRI运行时负责将CRI调用翻译为容器运行时（Containerd、CRI-O）的具体操作。CRI的标准实现：
- Containerd（Docker的后端运行时）：CNCF认证，最流行的CRI实现，广泛用于生产环境，轻量高效，架构灵活。
- CRI-O（Red Hat开发）：专为kubernates设计，精简高效，支持OCI兼容镜像，无守护进程，安全性高，广泛用于OpenShift平台。

OCI定义了容器运行时与镜像的标准化接口，而CRI是kubernetes与这些OCI标准容器运行时通信的统一接口，两者并非竞争关系，而是不同层面的表转化接口。OCI专注容器底层技术的标准化；CRI专注kubernates与容器运行时对接的标准化。

#### k8s & 集群组件

kubernetes通过抽象和统一管理容器，提供了可靠的分布式系统管理方案：
- Pod：kubernates最小的可调度单元，由一个或多个容器组成，共享存储和网络。
- Deployment：定义Pod的期望状态，控制Pod的创建、更新即扩缩容过程。
- ReplicaSet：Deployment的副本控制器，确保指定数量的Pod实例运行。
- Service：提供Pod的负载均衡和服务发现机制。
- Namespace：逻辑隔离多个资源和环境的命名空间，实现资源的隔离管理。

###### k8s集群组件与架构

kubernates集群由控制平面（controll plane）与节点（Node）两个主要部分构成，控制平面负责管理集群的状态和决策，节点则负责运行容器负载。

Master节点组件：是kubernates的控制平面，包含以下核心组件：
- API Server：是集群的统一入口，暴露Restful API，负责集群中所有资源的交互与管理，所有请求都要经过API Server。实现认证、授权、访问控制和资源管理的核心功能。与ETCD存储交互，维护集群状态。
- Scheduler(调度器)：Scheduler负责决定新创建的Pod应该部署到哪个节点。根据节点资源利用情况，Pod对资源的需求来做出决策。调度策略包括负载均衡、资源分配公平性、亲和性（Affinity）。Scheduler可配置和扩展，适应多样的调度需求。
- Controller Manager：负责确保集群实际状态始终与用户声明的期望状态保持一致，核心控制器包括：Node Controller：节点的故障检测与修复。ReplicaSet Controller：确保Pod副本数符合定义。Deployment Controller：负责滚动升级、扩缩容。Service Controller：与云提供商交互，管理负载均衡。
- ETCD存储服务： ETCD是kubernates集群状态存储组件，分布式键值存储，提供强一致性、高可用性。存储集群所有状态和配置信息。集群内的各组件均通过API Server访问ETCD来获取状态信息。

Node节点组件：是实际运行容器负载的机器，包括以下核心组件：
- kubelet：kubelet是kubernates节点上的主要管理代理，负责节点的Pod的生命周期管理。接收API Server的指令，管理Pod和容器。负责容器启动、停止、健康监测即状态上报。与容器运行时（containerd或CRI-O等）交互，实现容器的实际操作。
- kube-proxy：是节点上的网络代理，实现Pod网络通信和负载均衡。管理Service的网络规则，实现集群内外流量转发。支持IPtables和IPVS等多种负载均衡模式。保证网络通信的高效性和稳定性。
- 容器运行时：容器运行时负责实际运行容器。Containerd（最常见、轻量级，兼容OCI和CRI）。CRI-O（专为kubernates优化的运行时）、Docker。容器运行时通过CRI接口与kubelet通信，负责容器的创建、运行、停止和删除。

kubernates组件的交互流程：
- 用户通过kubectl命令发送请求的API Server。
- API Server验证请求并更新ETCD中的预期状态。
- scheduler从API Server获取Pod的调度请求，进行节点选择并更新Pod的状态。
- kubelet定期访问API Server获取节点的分配任务，，并调用容器运行时启动容器。
- Controller Manager定期对比集群的实际状态与期望状态，进行自动修正。
- Pod启动后，Kube-proxy根据服务定义设定网络规则，实现服务发现和负载均衡。

#### k8s的网络原理和通信机制

kubernates的网络模型主要实现了容器内通信，Pod间通信（同节点或跨节点），以及外部网络对Pod的访问。kubernates的网络模型有如下基本要求：集群内所有Pod间通信无障碍（同节点或跨节点）。每个Pod拥有独立的IP地址，且可直接访问。容器之间网络透明，容器无需关心网络实现细节。支持灵活的外部访问机制。kubernates网络模型基于第三方CNI插件实现，如：Flannel、Calico、Cilium等。

Pod内容器通信：Pod是kubernates的最小网络单元，同一个Pod中有多个容器共享网络命名空间，拥有相同的端口和IP地址。容器通过localhost直接通信，端口空间共享，不同容器监听端口不能重复。容器间使用本地loopback地址通信，无需经过任何网络设备，性能极高。

同节点Pod间通信流程：每个Pod都连接到同一个虚拟网桥（如docker0或cni0），虚拟网桥维护Pod间的转发规则，Pod间通信经虚拟网桥快速转发。Pod_A -> veth_A -> Bridge -> veth_B -> Pod_B，这种通信简单高效，延迟较低。

跨节点Pod通信：一来第三方CNI插件实现路由和封装，主要包括以下几种：
- Overlay网络（如Flannel、Weave）：通过隧道封装通信（如VXLAN）。示意流程：Pod_A -> Node_ABridge -> VXLAN -> Node_BBridge->Pod_B
- BGP路由网络（Calico）：利用BGP协议直接在网络中路由Pod IP地址，通信更直接，无封装开销。Pod_A -> Node_A-> BGP Routing -> Node_A -> Pod_B

CNI网络插件的原理和作用：kubernates网络接口规范为CNI，作用如下：在Pod创建于销毁时，为Pod提供和释放网络资源。定义标准接口，允许第三方插件实现网络功能，提供IP地址分配，路由管理和网络隔离等功能。CNI插件包括：Flannel：简单易用，基于Overlay网络、Calico：高性能，BGP路由实现，支持网络策略、Cilium：eBPF驱动的高效插件，支持微服务网络安全与观测。

外部访问Pod方法：
- NodePort Service：通过节点IP和指定端口暴露服务。External Client ->Node IP : Node IP -> Service -> Pod。简单易用，适合开发测试。
- LoadBalancer Service：云环境中自动创建负载均衡器，转发到Pod。 Client -> LoadBalancer -> Service -> Pod。生产环境广泛采用，但依赖云提供商支持。
- Ingress资源：基于域名和路径进行流量转发。Client -> IngressController ->Service -> Pod。更灵活，更便于维护和管理。

CNI插件的故障处理：当CNI插件或Pod网络发生故障时：Pod无法启动，状态为ContainCreating，节点间Pod无法通信。网络延迟，丢包显著增加。故障排查步骤：
- 检查节点CNI插件状态记日志：kubectl describe pod [pod_name]; journalctl -u kubelet。
- 检查Pod IP分配情况：kubectl get pods - o wide
- 检查网络接口状态：ip addr || ip route
- 重启节点网络组件或Pod：systemctl restart kubelet | kubectl delete pod [pod_name]

##### k8s存储技术与实现机制

kubernates存储模型旨在解决容器数据持久化的问题，核心概念包括：
- Volume（卷）：Pod级别的持久存储，生命周期随Pod存在。
- PersistentVolume（PV）：集群管理员配置的持久化存储资源。
- PersistentVolumeClaim（PVC）：Pod请求存储资源的一种声明。
- StorageClass（存储类）：描述不同存储类型和特性的配置模版。

Volume（卷）是kubernates最基础的存储抽象，特点为：Pod中的容器可以共享同一个Volume，支持多种存储后端（如本地存储、NFS和云存储）。Volume的生命周期与Pod一致，Pod删除之后，Volume被释放。PersistentVolume提供长期存储，不依赖于Pod的生命周期。kubernates常见卷类型：emptyDir：临时存储卷，Pod生命周琴内有效。hostPath：宿主机目录挂载到容器中。configMap、Secret：配置和秘钥管理。PersistentVolume（PV）：提供持久化数据存储。

持久化卷（PV）与PVC：
- PersistentVolume（PV）是集群管理员预先提供的一种持久化存储资源，生命周期独立于Pod。提供持久存储能力，可被不同的Pod使用，独立于特定Pod生命周期。可静态和动态创建。
- PersistentVol Claim(PVC)：代表应用对存储资源的需求，由用户创建。PVC定义存储需求，包括大小，访问模式等。kubernates复杂将PVC与合适的PV进行绑定。PV与PVC的交互流程如下：Pod-> PVC -> PV（有管理员提供或动态创建）

StorageClass（存储类）：用于动态创建和管理存储资源的模版。用户申请存储时指定StorageClass，kubernates根据StorageClass自动创建PV，允许集群管理员定义不同类型（如SSD、高性能磁盘、网络存储）的存储资源模版。

存储访问模式：kubernates存储卷支持多种访问模式，主要包括：ReadWriteOnce（RWO）：可被单个节点读写，但Pod数据存储，如数据库实例，状态化应用。ReadOnlyMany（ROX）：可被多个节点只读访问，静态内容（日志、数据分析）。ReadWriteMany（RWX）：可被多个节点读写，共享存储（如NFS、CephFS）。

kubernates的存储流程与实现原理：
- 用户定义PVC申请存储。
- Controller Manager根据StorageClass动态创建PV。
- PVC与PV自动绑定。
- Pod引用PVC实现存储卷挂载，容器启动时挂载卷到容器内部。
- 存储卷被Pod使用，数据在容器生命周期结束后仍保持持久化。

存储故障处理方法：
- 存储卷无法挂载。
- Pod状态为pending。
- 数据读写失败，延迟显著。

排查步骤：检查PV和PVC状态：kubetctlget pv,pvc。检查节点挂载情况：mount | grep pvc , dmesg | grep mount。检查存储后端状态（如NFS、Ceph、云存储状态）。检查相关kubernates日志（如kubelet、controller -manager）

#### 云原生架构设计与最佳实践

云原生架构的核心设计原则：
- 服务化与微服务：拆分粒度：业务按领域驱动设计（DDD）划分微服务，避免“分布式单体”。自治性：每个服务独立开发、部署、扩展，通过API（REST/gRPC）通信。服务治理：熔断、限流、重试（如Hystrix、Resilience4j）保障稳定性。
- 容器化与不可变基础设施：容器镜像：通过Docker将应用与依赖打包，确保环境一致性。不可变性：运行时禁止直接修改容器，更新时替换新镜像（Immutable Infrastructure）。
- 动态编排与弹性伸缩：Kubernetes：自动化部署、扩缩容（HPA/VPA）、自愈（Pod健康检查）。Serverless：按需分配资源（如AWS Lambda），极致弹性。
- 声明式API与自动化：基础设施即代码（IaC）：用Terraform、Ansible定义资源。GitOps：通过Git仓库管理配置，ArgoCD实现持续同步。
- 可观测性（Observability）：Metrics：Prometheus采集指标，Grafana可视化。Logging：集中式日志（ELK、Loki）。Tracing：分布式链路追踪（Jaeger、Zipkin）。

云原生关键技术栈：
- 容器运行时：Docker/Containerd：标准化容器运行环境。Rootless容器：提升安全性。
- 编排与调度：Kubernetes：核心组件（API Server、etcd、kubelet）。多集群管理：Karmada、Clusternet。
- 服务网格（Service Mesh）：Istio：流量管理（金丝雀发布）、安全（mTLS）、可观测性。Linkerd：轻量级Mesh，适合低延迟场景。
- 持续交付（CI/CD）：Pipeline工具：Jenkins、GitLab CI、Tekton。镜像安全：Trivy扫描漏洞，Harbor私有仓库。
- 存储与网络：云原生存储：CSI驱动（Longhorn、Rook）。网络策略：Calico实现网络隔离（NetworkPolicy）。

最佳实践与落地策略：
- 微服务拆分与治理：拆分原则：单一职责（Single Responsibility）。独立数据库（每个服务对应独立DB或Schema）。异步通信（消息队列解耦，如Kafka、RabbitMQ）。API网关：路由、鉴权（OAuth2/JWT）、限流（Nginx/APISIX）。服务注册与发现：Consul、Eureka、Kubernetes Service。
- 弹性与高可用设计：容错机制：超时控制（客户端/服务端）。熔断降级（Sentinel、Istio Circuit Breaker）。多活架构：跨可用区（AZ）部署，避免单点故障。数据多副本（如Cassandra多数据中心复制）
- 安全最佳实践：零信任网络：服务间mTLS（双向认证）。网络策略（Kubernetes NetworkPolicy）。权限最小化：RBAC（基于角色的访问控制）。安全上下文（Pod Security Policies）。密钥管理：Vault、KMS加密敏感数据。
- 性能优化：资源配额：限制CPU/Memory（Requests/Limits）。避免资源争抢（QoS分级）。冷启动优化：预热Pod（Kubernetes Readiness Probe）。Serverless预留实例（如AWS Provisioned Concurrency）。
- 成本管理：自动扩缩容：HPA（基于CPU/内存）、KEDA（基于事件驱动）。Spot实例利用：混合使用按需实例和竞价实例（AWS Spot Fleet）。

典型云原生架构案例：
- 架构：服务网格：Linkerd保障低延迟通信。数据库：TiDB（分布式NewSQL）。安全：Vault管理密钥，mTLS加密通信。关键实践：多活部署（两地三中心）。每日执行混沌工程（Chaos Mesh）测试容错。

常见挑战与解决方案：
- 分布式事务一致性：Saga模式：通过补偿事务实现最终一致性。TCC（Try-Confirm-Cancel）：业务层两阶段提交。
- 配置管理：ConfigMap/Secrets：Kubernetes原生配置管理。外部化配置：Spring Cloud Config、Nacos。
- 监控告警：Prometheus AlertManager：设置阈值告警（如CPU >80%持续5分钟）。SLO/SLI：定义服务等级目标（如99.9%可用性）。
- 技术债务：渐进式迁移：单体应用逐步拆分为微服务（Strangler Fig模式）。自动化测试：API契约测试（Pact）、混沌测试（Chaos Monkey）。

未来趋势：
- 边缘计算：Kubernetes K3s、OpenYurt支持边缘节点。
- WebAssembly（Wasm）：轻量级沙箱，替代容器运行时（如WasmEdge）。
- AI/ML与云原生结合：Kubeflow管理机器学习流水线。

##### Service Mesh（Istio/Linkerd）架构设计与最佳实践？

首先，用户可能已经知道服务网格的基本概念，但需要更详细的架构设计。他们可能想知道Istio和Linkerd的核心组件有什么不同，各自的优缺点是什么。例如，Istio的功能更全面但复杂，而Linkerd更轻量，适合简单场景。用户可能面临选择困难，需要根据项目规模、团队经验等因素来决定。

Service Mesh 核心架构设计：
- 核心组件：数据平面（Data Plane）：Sidecar Proxy：每个服务实例旁挂载的轻量级代理（如Envoy/Istio-Proxy、Linkerd2-proxy），负责流量拦截与处理。功能：负载均衡、TLS加密、熔断、重试、指标收集。控制平面（Control Plane）：Istio：由Pilot（流量管理）、Citadel（证书管理）、Galley（配置管理）、Istiod（整合组件）构成。Linkerd：由Destination（服务发现）、Identity（mTLS）、Proxy Injector（自动注入Sidecar）构成。
- 架构模式：透明劫持：通过iptables或eBPF劫持Pod流量到Sidecar，业务代码无感知。声明式配置：通过YAML定义流量规则（VirtualService、DestinationRule），控制平面下发到数据平面。
- Istio 与 Linkerd 对比：
特性	Istio	Linkerd
代理类型	Envoy（高性能，高资源占用）	Linkerd2-proxy（Rust编写，轻量）
学习曲线	复杂（功能丰富）	简单（专注核心场景）
适用场景	大规模企业级微服务	中小规模、低延迟需求
扩展性	支持Wasm插件、自定义Envoy Filter	轻量级扩展
社区生态	CNCF毕业项目，生态完善	CNCF孵化项目，社区活跃

Service Mesh 核心功能实践：
- 流量管理：金丝雀发布：故障注入：模拟服务延迟或错误（测试系统容错性）。重试与超时：配置HTTP请求重试次数及超时阈值。
```yaml
# Istio VirtualService 示例
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: reviews
spec:
  hosts:
  - reviews
  http:
  - route:
    - destination:
        host: reviews
        subset: v1
      weight: 90%
    - destination:
        host: reviews
        subset: v2
      weight: 10%
```
- 安全：mTLS（双向认证）：Istio：通过PeerAuthentication启用全局或命名空间级mTLS。Linkerd：自动为所有服务通信启用mTLS，无需手动配置。细粒度授权：
- 可观测性：指标采集：Istio：集成Prometheus，暴露istio_requests_total等指标。Linkerd：内置Dashboard，支持Grafana集成。分布式追踪：通过Jaeger/Zipkin追踪请求链路（需在应用层传播Trace Header）。日志聚合：Sidecar代理日志输出到Fluentd/ELK。

最佳实践与落地策略：
- 渐进式采用：按命名空间启用Mesh：初期在非核心业务命名空间试点。Sidecar自动注入：
```yaml
# Kubernetes 注解示例
apiVersion: v1
kind: Namespace
metadata:
  name: my-app
  labels:
    istio-injection: enabled  # Istio自动注入
    linkerd.io/inject: enabled  # Linkerd自动注入
```
- 性能优化：Sidecar资源限制：
```yaml
# Istio Sidecar资源配额
resources:
  limits:
    cpu: 500m
    memory: 512Mi
  requests:
    cpu: 100m
    memory: 128Mi
```
- 代理调优：减少Envoy日志级别（--log-level warn）。Linkerd禁用不需要的TLS证书轮换（默认开启）。
- 网络策略：零信任网络：使用Kubernetes NetworkPolicy限制Pod间通信。Istio的AuthorizationPolicy与Linkerd的Server资源结合使用。
- 与现有系统集成：混合部署：Mesh与非Mesh服务共存时，通过ServiceEntry（Istio）或ExternalEndpoint（Linkerd）接入外部服务。API网关整合：将Istio Ingress Gateway或Linkerd的Gateway作为API网关，统一入口流量。

常见问题与解决方案：
- Sidecar启动顺序问题：症状：主容器先于Sidecar启动导致流量丢失。解决：使用holdApplicationUntilProxyStarts: true（Istio）或waitBeforeExitSeconds（Linkerd）延迟主容器启动。
- mTLS证书过期：预防：Istio：定期检查istioctl experimental wait验证证书状态。Linkerd：自动轮换证书，无需人工干预。
- 性能瓶颈：场景：Sidecar代理引入额外延迟。优化：启用Protocol Sniffing（Istio）自动检测HTTP/1.1、HTTP/2、gRPC。Linkerd默认使用HTTP/2多路复用减少连接数。

典型场景案例：
- 案例1：Istio实现灰度发布：步骤：部署v1和v2版本的服务，通过DestinationRule定义子集。配置VirtualService逐步调整流量权重。监控指标（错误率、延迟），确认稳定后切流。
- 案例2：Linkerd保障金融系统低延迟：实践：使用Linkerd的Service Profile定义请求重试策略。通过内置Dashboard实时监控P99延迟，快速定位瓶颈。

未来趋势：
- eBPF优化：Cilium Mesh替代传统Sidecar模式，减少资源开销。
- Wasm扩展：Istio支持Envoy Wasm Filter，动态加载安全/治理逻辑。
- 多集群Mesh：Istio的Multi-Cluster Service与Linkerd的Multi-Cluster Gateways实现跨云治理。

Istio适合复杂场景下的精细化治理，需权衡其资源消耗与功能丰富性。Linkerd以轻量、易用见长，适合快速落地核心功能。核心原则：按需启用Mesh功能，避免过度设计，结合业务需求选择技术栈，持续监控与调优。

##### 有状态服务（数据库&中间件）扩缩容架构设计和最佳实践？

扩缩容通常包括垂直扩展（增加单个节点的资源）和水平扩展（增加节点数量）。对于数据库，这可能涉及读写分离、分库分表、主从复制等；中间件如消息队列、缓存可能需要分区、副本机制等。用户可能关心如何设计这些系统以支持自动扩缩容，同时保证数据一致性、高可用性和性能。需要考虑不同数据库类型（关系型、NoSQL）和中间件（如Kafka、Redis、RabbitMQ）的扩缩容策略。例如，关系型数据库如MySQL的扩展可能比较困难，而NoSQL如Cassandra天生支持分布式扩展。中间件如Kafka通过分区和副本实现扩展，Redis则可以通过分片或集群模式。我还需要考虑到云原生的环境，比如在Kubernetes中部署数据库和中间件的情况，使用StatefulSet管理有状态应用，利用Operator模式自动化扩缩容操作。同时，结合监控和自动化工具（如Prometheus、Horizontal Pod Autoscaler）来实现动态扩缩容。另外，用户可能对成本优化感兴趣，比如按需扩缩容以节省资源，使用Spot实例等。同时，数据备份和恢复策略在扩缩容过程中也很重要，确保数据不丢失。还要注意不同中间件的特性，比如Kafka分区数量的调整可能比较麻烦，需要预先规划足够的扩展性；Redis集群在扩缩容时可能需要resharding，这会影响性能，需要分步骤进行。

数据库和中间件的扩缩容是构建高可用、弹性伸缩系统的核心能力，尤其在云原生环境下，需结合业务场景设计合理的水平/垂直扩展策略。以下是针对不同数据库类型（关系型、NoSQL）及中间件（消息队列、缓存等）的架构设计与最佳实践。
- 通用扩缩容原则：分层设计：计算与存储分离：通过解耦计算节点（如数据库实例）与存储层（如云盘、对象存储），支持独立扩展。代理层：使用代理（如ProxySQL、Redis Sentinel）隐藏后端节点变化，客户端无感知扩缩容。
- 自动化与监控：指标驱动：基于CPU、内存、连接数、QPS/TPS等指标触发扩缩容（如Prometheus + Kubernetes HPA）。自愈机制：自动替换故障节点（如Kafka Controller选举、MySQL主从切换）。
- 数据一致性保障：最终一致性：NoSQL数据库（如Cassandra）通过Quorum机制平衡一致性与可用性。强一致性：关系型数据库（如MySQL Group Replication）使用Paxos/Raft协议保证强一致。
- 成本优化：按需扩缩：结合业务峰谷时段动态调整资源（如AWS Aurora Serverless）。冷热分离：历史数据归档至低成本存储（如S3 Glacier）。

数据库扩缩容设计：
- 关系型数据库（MySQL/PostgreSQL）：水平扩展（分库分表）：分片策略：按业务键（如用户ID）哈希分片，使用ShardingSphere、Vitess管理路由。读写分离：主库写，从库读（通过ProxySQL自动路由）。垂直扩展：升级实例规格（CPU/内存），需停机或在线DDL（如MySQL 8.0 Instant ADD COLUMN）。云原生方案：AWS RDS/Aurora：支持自动扩展存储和只读副本。Kubernetes Operator：使用Kubernetes Operator（如Percona Operator）自动化管理集群。
- 最佳实践：分片预规划：预估数据增长，避免后期分片迁移（如每个分片预留50%容量）。在线DDL工具：使用gh-ost或pt-online-schema-change避免锁表。

NoSQL数据库（MongoDB/Cassandra/Redis）：
- MongoDB：分片集群：通过mongos路由查询，分片键选择高基数字段（如时间戳+设备ID）。副本集：自动故障转移，最多50个成员。
- Cassandra：一致性级别：QUORUM（读+写副本数 > 总副本数）。扩缩容步骤：添加新节点到环（Token Ring）。运行nodetool repair同步数据。
- Redis：Cluster模式：16384个哈希槽分片，支持动态增删节点。Proxy方案：使用Twemproxy或Redis Cluster Proxy隐藏分片细节。
- 最佳实践：避免热点分片：Cassandra使用RandomPartitioner，Redis使用HASH_SLOT分散数据。扩缩容窗口：选择低峰期执行，Cassandra建议单次扩容不超过25%节点。

中间件扩缩容设计（消息队列（Kafka））：
- Kafka：分区扩容：增加Topic分区数，需重启Producer/Consumer或使用kafka-reassign-partitions工具。Broker扩展：新Broker自动加入集群，分区副本自动均衡。
- 最佳实践：分区预分配：Kafka分区数建议为Broker数量的整数倍，避免数据倾斜。消费者组管理：Kafka消费者数量与分区数匹配，避免资源浪费。

中间件扩缩容设计（缓存（Redis））：
- Redis Cluster：扩缩容流程：添加新节点，分配空哈希槽。迁移槽位：redis-cli --cluster reshard。删除旧节点，槽位重新分配。云托管服务：AWS ElastiCache：支持自动分片（Sharding）和副本扩展。
- 最佳实践：数据预热：新节点加入前预加载热点数据，避免缓存击穿。多级缓存：本地缓存（Caffeine）+ 分布式缓存（Redis）减少网络开销。

API网关与负载均衡器：
- 动态扩缩容：Nginx/HAProxy：通过Kubernetes Ingress Controller自动扩展Pod副本。Envoy：支持动态配置（xDS API），无需重启。

云原生扩缩容实践：
- Kubernetes StatefulSet管理有状态服务：有状态服务扩缩容：数据库/中间件使用StatefulSet保障Pod唯一标识（如MySQL StatefulSet mysql-0、mysql-1）。持久化存储（PVC）随Pod自动绑定。Operator模式：使用Prometheus Operator、Redis Operator自动化扩缩容与故障恢复。
- Serverless数据库：AWS Aurora Serverless：根据负载自动调整ACU（Aurora Capacity Units）。Google Cloud Spanner：全球级水平扩展，无需手动分片。

典型问题与解决方案：
- 数据迁移延迟：方案：双写过渡（新旧集群同时写入），增量数据同步完成后切换读流量。
- 扩缩容期间性能抖动：方案：限流（如Kafka副本同步限速）、分批次操作。
- 分布式事务一致性：方案：Saga模式补偿事务，或使用支持分布式事务的数据库（如TiDB）。

案例参考：
- 案例1：电商大促Redis集群扩容：场景：QPS从10万突增至100万。操作：提前扩容：增加Redis分片数，预热热点商品数据。流量切换：通过Proxy将新请求路由至新分片。监控：实时跟踪缓存命中率与延迟（如Grafana看板）。
- 案例2：Kafka分区扩容，场景：Topic吞吐量不足导致消息积压。步骤：修改Topic分区数（从100增至200）使用kafka-reassign-partitions重新分配分区至新增Broker。调整Producer分区策略（如RoundRobin）。

总结：数据库扩缩容核心：分片设计、读写分离、数据一致性保障。中间件扩缩容核心：分区/副本管理、流量路由、资源隔离。云原生最佳实践：优先使用托管服务（如RDS、ElastiCache），结合Kubernetes Operator实现自动化。关键原则：预规划容量、渐进式变更、监控驱动决策。

##### HPA（基于CPU/内存）、KEDA（基于事件驱动）技术实现细节

在 Kubernetes 中，自动扩展（Auto-scaling）是确保应用程序在不同负载条件下保持性能和可用性的关键功能。Horizontal Pod Autoscaler (HPA) 和 Kubernetes Event-Driven Autoscaling (KEDA) 是两种常用的自动扩展技术，分别基于 CPU/内存和事件驱动。

Horizontal Pod Autoscaler (HPA)：HPA 是 Kubernetes 的内置自动扩展器，基于 CPU 使用率或其他资源指标（如内存）来调整 Pod 的数量。实现细节：
- 配置 HPA：使用 kubectl autoscale 命令或 YAML 文件来配置 HPA。指定目标资源（如 Deployment、ReplicaSet 或 StatefulSet）和目标指标（如 CPU 使用率）。
```yaml
apiVersion: autoscaling/v2beta2
kind: HorizontalPodAutoscaler
metadata:
  name: my-app-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: my-app
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 50
```
- 工作原理：HPA 控制器定期从资源指标 API 获取目标 Pod 的当前指标。根据当前指标和目标指标计算所需的副本数量。调整目标资源的副本数量以匹配计算出的值。
- 优点：简单易用，适用于基于资源使用率的扩展需求。与 Kubernetes 集群紧密集成，无需额外组件。
- 缺点：只能基于资源指标进行扩展，无法处理自定义指标或事件驱动的扩展需求。

Kubernetes Event-Driven Autoscaling (KEDA)：KEDA 是一个基于事件驱动的自动扩展器，可以根据各种事件源（如消息队列、数据库等）动态调整 Pod 的数量。实现细节：
- 配置 KEDA：安装 KEDA 控制器和指标服务器。创建 ScaledObject 自定义资源，定义扩展策略和事件源。
```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: my-app-scaledobject
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: my-app
  minReplicaCount: 1
  maxReplicaCount: 10
  triggers:
  - type: azure-queue
    metadata:
      queueName: my-queue
      queueLength: "5"
```
- 工作原理：KEDA 控制器监控配置的事件源，获取当前事件数量或负载。根据事件源的负载计算所需的副本数量。调整目标资源的副本数量以匹配计算出的值。
- 优点：支持多种事件源，适用于事件驱动架构。可以处理自定义指标和复杂的扩展需求。
- 缺点：需要额外安装和配置 KEDA 组件。可能需要与特定的事件源集成，增加了复杂性。

##### SLO/SLI

SLO（Service Level Objective）和SLI（Service Level Indicator）是云原生领域中用于衡量服务质量和可靠性的两个重要概念。
SLI（服务等级指标）：SLI是用于衡量服务健康状况的具体指标。这些指标通常与服务的**可用性、延迟、吞吐率**和**成功率**等方面有关。SLI的选择取决于需要观测的服务维度以及现有的监控手段。常见的SLI包括：
- 可用性：服务成功响应的时间比例。
- 延迟时间：服务返回请求的响应所需时间。
- 吞吐率：服务处理请求的速率，如每秒请求数（QPS）。

SLO（服务等级目标）:SLO是基于SLI定义的目标值或范围值，用于描述服务在一段时间内达到某个SLI指标的比例。SLO提供了一种形式化的方式来描述、衡量和监控微服务应用程序的性能、质量和可靠性。例如：
- 99%的请求延迟小于500毫秒。
- 每分钟平均QPS大于100,000/s。
- SLO的设定有助于服务提供者与客户之间建立明确的服务质量期望。

SLO与SLI的关系：SLO是基于SLI来定义的。通过设置SLO，服务提供者可以明确服务的预期状态，而SLI则提供了衡量这些预期的具体指标。SLO的达成情况通常通过监控SLI来评估。实践中的应用：在实践中，SLO和SLI被广泛应用于云原生系统的可靠性和性能监控。通过使用Prometheus等监控工具，可以实时计算SLI，并根据SLO设定告警和自动化操作，以确保服务的可靠性和性能。

与SLA的区别：**SLA（服务等级协议）**是基于SLO衍生出来的协议，通常用于服务提供商与客户之间的合同中，明确了如果SLO未达标时的赔偿或处罚条款。SLO是服务提供者内部用于管理和优化服务质量的目标，而SLA则是对外的承诺。总之，SLI提供了服务质量的可观测性，SLO则是基于SLI设定的具体目标，而SLA是对SLO达成情况的法律层面承诺。

##### Cilium

Cilium是一个开源项目，旨在为Kubernates集群和其它容器编排平台等云原生环境提供网络、安全性、可观察性。Cilium的基础是一项名为eBPF的新Linux内核技术，该技术能够将强大的安全性、可见性和网络控制逻辑动态的插入到Linux内核当中，eBPF提供高性能网络、多集群和多云能力、高级负载均衡、透明加密、网络安全能力和可观察性等。由于eBPF在Linux内核中运行，因此无需更改应用程序代码或容器配置即可应用和更新Cilium安全策略。

Hubble是一个完全分布式的网络和安全可观测性平台。它建立在Cilium和eBPF之上。能够以完全透明的方式深入了解服务以及网络基础设施的通信和行为。通过基于Cilium的构建，Hubble可以利用eBPF实现可视化，所有可视化都可编程的，并允许采用动态的方法，以最大限度的减少开销，同时根据用户的要求提供深入而详细的可视化。
- 服务依赖关系与通信图：哪些服务正在相互通信？通信频率如何？服务以来关系图如何？正在进行哪些HTTP调用？服务从哪些Kafka主题消费或生成那些内容？
- 网络监控与报警：是否有网络通信失败？通信失败的原因是什么？是应用程序的问题还是网络问题？通信在第4层(TCP)还是在第7层(HTTP)中断？过去5分钟内哪些服务遇到DNS解析问题？那些服务最近遇到过TCP连接中断或连接超时？未答复的TCP SYN请求的比率是多少？
- 应用程序监控：特定服务或所有集群的5xx或4xx HTTP相应代码的发生率是多少？集群中HTTP请求和响应的TP95和TP99延迟是多少？哪些服务表现最差？两个服务之间的延迟是多少？
- 安全与可观察性：由于网络策略，哪些服务的连接被阻止？那些服务已从集群外部访问？那些服务已解析特定的DNS名称？

eBPF能够以前所未有精细度和效率实现对系统和应用程序的可见性和控制，它以完全透明的方式实现这一点，无需对应用程序进行任何更改。eBPF同样能够处理容器工作负载。例如虚拟机和标准Linux进程。现在采用微服务架构，大型应用程序被拆分为小的独立服务，这些服务使用HTTP等轻量级协议进行相互通信，微服务往往具有动态性，随着应用程序扩大/缩小以适应负载变化，每个容器都会启动或者销毁。传统的Linux网络安全的方法（例如iptables）会过滤IP地址和TCP/UDP端口，但IP地址在动态的微服务环境中经常变动。容器的生命周期不稳定，导致这些方法难以与应用程序并行扩展，因为负载平衡表和访问控制列表承载着数十万条规则，这些规则需要不断地增长并且频繁更新。协议端口无法在用于区分应用程序的流量，因为该端口用于跨服务的各种消息。其他的挑战是提供准确的可见性，因为在传统中IP地址作为主要的识别工具，而在微服务架构中，其生命周期可能缩短至秒级。通过利用Linux eBPF，Cilium提供了透明插入安全可见性 + 执行的能力但这样做的方式是基于服务/pod/容器身份（与传统系统IP地址标识不同），并可以在应用层（如HTTP）上进行过滤。因此Cilium不仅将安全性与寻址分离，在高度动态的环境中使用安全策略变得简单，还可以通过在应用层运行来提供更强大的安全隔离。
- 透明地保护和保障API安全：能够保障应用层协议的安全，例如REST/HTTP、gRPC和Kafka，传统防火墙在第3层和第4层运行，在特定端口上运行的协议要么完全被信任，要么完全被阻止。Cilium提供对单个应用层协议请求进行过滤的能力。例如：允许所有HTTP请求使用GET方法和路径/public/.*，拒绝所有其他请求。允许service1在Kafka主题topic1上生产消息，service2在topic1上消费消息。拒绝所有其他Kafka消息。要求在所有REST调用中存在header X-Token: [0-9]+。
- 基于身份的服务间通信安全：分布式应用依赖于容器等技术，以提高部署的灵活性和按需扩展，这导致在短时间内启动了大量容器，典型的容器防火墙通过过滤源IP地址和目标端口来保障工作负载的安全，每当集群中启动一个容器，需要操作所有服务器上的防火墙。为了避免这种限制，Cilium共享了相同的安全策略，并为容器组分配了一个安全身份，该身份与容器发出的所有网络数据包相关联。从而可以在接收节点验证身份，安全身份通过键值存储来管理。
- 保障队外部服务的访问安全：基于标签的安全是集群内部访问控制的首选工具。为了保障对外部服务de访问安全，支持传统的基于CIDR的安全策略，用于入站和出站流量，可以限制容器对特定IP范围的访问。
- 负载均衡：Cilium实现了容器之间以及与外部服务之间的分布式负载均衡，能够完全替代诸如kube-proxy之类的组件，负载均衡是使用高效的哈希表在eBPF中实现，支持无线扩展。对于南北向负载均衡，Cilium的eBPF进过优化已达到最大性能，可以附加到XDP，并支持服务器返回（DSR）以及Maglev一致性哈希，如果负载均衡操作不在源主机上执行，，对于东西向负载均衡，Cilium在Linux内核的套接字层（例如TCP连接时）执行高效的服务到后端的转换，从而避免在较低层次上进行每个数据包的NAT操作开销。
- 带宽管理：Cilium通过高效的EDT（最早离开时间）的速率限制和eBPF实现了对节点出口的容器流量的带宽管理，这可以显著减少应用程序的传输尾部延迟，并在多队列网卡下避免锁定问题相比于传统的HTB（层令牌桶）或TBF（令牌桶过滤器），例如在带宽CNI插件中使用的方法。
- 监控与故障排除：获得可见性和排除故障的能力是分布式系统运行的基础。故障排除工具有：1、带有元数据的事件监控，当数据包被丢弃时，工具不仅报告数据包的源和目标IP，还提供发送方和接收方的完整标签信息以及其他信息。2、通过Prometheus导出的度量，关键度量通过Prometheus导出，以便与现有的仪表板集成。3、Hubble，专为Cilium编写的可观测性平台。它提供服务依赖关系图、运营监控和报警，以及基于流日志的应用和安全可见性。

Cilium组件：
- Cilium代理(Cilium Agent)：在集群中的每个节点上运行，从更高维度来看，Cilium Agent通过监听Kubernates事件来了解容器或工作负载的启动和停止时间，同步集群状态。Cilium Agent负载管理Linux内核中的eBPF程序，这些程序用于控制进出容器的所有网络访问。Cilium Agent根据配置的网络策略和安全规则生成eBPF程序，并将其加载到内核中。Cilium Agent还负责服务发现和负载均衡，取代传统的kube-proxy。
- Cilium CLI： 是一个与Cilium Agent一起安装的命令行工具，它允许用户通过REST API与同节点上的本地Cilium Agent进行通信，从而检查Cilium Agent的状态，确保其正常运行。主要功能：1、状态检查，通过Cilium CLI，可以检查本地Cilium Agent的状态，确保其运行正常；2、eBPF地图访问，直接访问eBPF地图内容，以便随时验证网络状态和配置；3、命令支持，提供了多种命令用于管理和调试Cilium的各个组件，如endpoint、policy、metrics等。
- Cilium Operator：是Cilium 网络插件的管理平面组件，负责执行集群级别的运维任务，确保Cilium 网络功能的高效运行。Cilium Operator的核心功能：1、集群级IP地址管理（IPAM），当Cilium运行在CRD模式或云提供商模式（如Azure、AWS）时，Operator负责从Kubernetes的Node资源中获取Pod CIDR，并同步到CiliumNode资源中。在负载均衡IP管理场景中，Operator为type: LoadBalancer的Kubernates服务分配和管理IP地址。2、CRD注册与同步，Operator自动注册Cilium所需的自定义资源（CRD），如CiliumBGPAdvertisement、CiliumNode等，用于定义网络策略、BGP配置等。3、BGP配置和网络宣告，网络宣告（Network Announcement）通常指的是在网络中广播或发布网络可达性信息的过程，确保网络设备之间能够相互通信。配合cilium-bird组件，Operator负责将Kubernates集群内的Pod IP通过BGP协议宣告到外部网络（如物理交换机），实现跨网络的路由可达。4、垃圾回收和资源清理，清理孤儿资源：例如删除已终止Pod对应的CiliumEndpoint对象，或清理无效的CiliumNode资源。定期同步KVStore（如ETCD）中的心跳信息，确保集群状态一致性。5、网络策略派生与转换，将高级网络策略（如基于云服务标签的toGroups规则）转换为具体的Cilium网络策略(CNP/CCNP)。6、Ingress/Gateway API支持，解析Kubernates的Ingress或Gateway API对象，生成对应的CiliumEnvoyConfig配置，并同步Secret到Cilium管理的命名空间。高可用性（HA）设计：Cilium Operator通过Kubernates的Leader选举机制实现多副本高可用，仅Leader实例执行关键任务（如CIDR分配），其余副本处于备用状态。与Cilium Agent相比，Cilium Agent：职责范围属于节点级任务，eBPF程序加载、Pod流量策略执行，主要运行在每个节点之上，处理实时数据面操作。Cilium Operator：职责范围是集群级任务，IP分配、CRD管理、BGP配置，全局管理，不参与数据转发决策。故障影响：即使Cilium Operator短暂不可用，集群仍能继续运行，但以下操作可能会延迟：新Pod的IP地址分配、节点加入时的CIDR分配、KVStore心跳更新失败可能导致Agent异常重启。Cilium Operator是Cilium生态中负责**全局状态管理**的核心组件，通过解耦集群级任务与节点级操作，提升了系统的可靠性和扩展性。
- CNI插件：是一种用于配置Linux容器网络接口的框架。它提供了一种标准化的方式来管理容器网络，允许不同插件实现不同的网络功能。CNI插件在Kubernates中尤其重要，因为它帮助配置和管理Pod之间的网络连接。CNI的插件类型：接口插件（Interface Plugins），这些插件负责在容器中创建网络接口，并确保容器与网络的连接。链式插件（Chained Plugins）：这些插件可以调整已经存在的接口配置，可能需要创建额外的接口来实现特定的网络配置。CNI 插件的工作原理：容器运行时请求网络设置，当容器创建时，容器运行时会调用CNI插件来配置网络。CNI插件设置网络环境，插件根据配置信息为容器分配IP地址，配置网络接口，并设置必要的路由规则。网络连接，配置完成后，容器可以通过其IP地址与其它容器或外部网络进行通信。流行的CNI插件：Calico，提供高可扩展性和网络策略执行功能，支持 BGP 路由等。Flannel：轻量级的网络解决方案，支持多种后端机制。Weave Net：提供灵活的网络解决方案，创建网状覆盖网络连接集群中的所有节点。Cilium：使用 eBPF 实现高性能的网络和安全功能，支持细粒度的网络策略。Multus：允许在单个 Pod 中附加多个网络接口，适用于复杂的网络场景。CNI插件支持以下几种操作：ADD：添加容器到网络或应用修改。DEL：从网络中删除容器。CHECK：检查容器的网络配置。GC：垃圾回收，用于清理不再使用的资源。VERSION：显示插件版本。

Hubble架构：
Hubble 是一个完全分布式的网络和安全可观测性平台，建立在 Cilium 和 eBPF 之上。它提供了对服务通信和网络基础设施的深入可见性，支持在节点、集群或多集群环境中进行监控。
- Hubble服务器：Hubble服务器在每个节点上运行，负责从Cilium检索基于eBPF的可视性数据，他被嵌入到了Cilium Agent中，已实现高性能和低开销。接口：提供gRPC服务来检索流量和Prometheus指标。
- Hubble Relay（中继）：Hubble Relay是一个独立的组件，充当集群中所有Hubble服务器的中介。它通过连接到每个Hubble服务器的gRPC API，提供集群范围内的可视性。
- Hubble CLI & UI：Hubble CLI是命令行工具，用于连接Hubble Relay的gRPC API或本地服务器，以检索流量事件。Hubble UI是通行用户界面，利用基于Hubble Relay的可视性，提供服务依赖性和连接图的可视化。

架构流程：1、数据收集，Cilium Agent通过eBPF收集网络数据， 并将其发送给Hubble服务器。2、数据处理，Hubble服务器对接收到的数据进行聚合、分析和存储。3、集群范围可视性，Hubble Relay提供集群范围内的可视性，通过连接所有Hubble服务器的gRPC API。4、用户交互，用户通过Hubble CLI & UI与Hubble Observer服务交互，以获取网络流量信息。

eBPF原理：
eBPF是一种Linux内核技术，允许开发者在内核空间中运行沙盒程序，而无需修改内核源代码或加载额外的内核模块。它通过安全、非侵入的方式扩展操作系统功能，为网络、性能检测、安全性等领域提供强大的支持。
- 事件驱动：eBPF程序是事件驱动的依附于内核代码路径中的特定触发点（称为”钩子“）这些钩子会在特定事件发生时触发eBPF程序运行。常见的钩子包括：系统调用（如进程创建等）、函数入口和出口、网络事件（如数据报接收）、内核探针（kprobes）和用户探针（uprobes）。
- 编译与验证：eBPF通常使用受限的C语言编写，并通过工具链（如LLVM或CLang）编译为eBPF字节码。在加载到内核之前，字节码会经过验证器检查，以确保程序不会执行非法操作（如无限循环和越界访问），只有通过验证的程序才能被加载到内核。
- 运行与数据交互：验证通过之后，eBPF字节码会被加载到内核，并附加到指定的钩子上。当事件被触发时，eBPF程序在内核中运行。程序可以调用预定义的辅助函数(help functions)，用于访问内存或操作数据。eBPF使用maps作为数据结构，用于在用户空间和内核空间之间共享数据。这些映射以键值对形式存储，可以保持状态或传递信息。
- 卸载程序：当eBPF程序完成任务后，可以通过系统调用将其卸载，从而释放资源。

eBPF的优势：
- 安全性：eBPF程序运行在沙盒中，经过严格验证，不会破坏系统稳定性。
- 性能高效：直接在内核中运行，避免了频繁的用户空间与内核空间交互减少了性能开销。
- 灵活性：支持动态加载程序，无需重启或修改内核代码。
- 扩展性：不仅限于网络领域，还可用于性能监控、动态追踪、安全策略等。

Cilium 需要一个数据存储使Cilium Agent之间传播状态。它支持以下几种数据存储：
- Kubernetes CRDs（默认）：默认选择是Kubernates自定义资源定义（CRDs）来存储数据并传播状态。CRDs有Kubernates提供。CRDs是 Kubernetes 的原生机制，易于管理和集成。
- 键值存储：所有状态存储和传播的要求都可以通过Kubernates CRDs来满足。键值存储可以作为可选项使用，已优化集群的扩展性，因为直接使用键值存储可以更高效地处理更改通知和存储需求。etcd：一种流行的分布式键值存储，提供高可用性和一致性。Cilium 默认使用Kubernetes CRDs作为数据存储，但也支持使用键值存储（如ETCD）来提高集群的可扩展性和性能。

术语：
- 标签：标签是一种通用、灵活且高度可扩展的方式，用于处理大量资源，因为它们允许任意分组和集合的创建。每当需要描述、寻址或选择某些内容时，都是基于标签来进行的。端点(Endpoint)会根据容器运行时、编排系统或其他来源分配标签。网络策略(Network Policy)根据标签选择允许通信的端点对。这些策略本身也通过标签来识别。标签是由键和值组成的字符串对，标签可以格式化为单个字符串，格式为key=value。键部分是必须的，并且必须是唯一的。使用反向域名来实现的，例如：io.cilium.mykey=myvalue，值部分是可选的，可以省略，例如 io.cilium.mykey。键名通常应由字符集 [a-z0-9-.] 组成。在使用标签选择资源时，键和值都必须匹配。例如，如果一个策略应用于所有带有标签my.corp.foo的端点，那么标签my.corp.foo = bar将不匹配选择器。
- 标签来源：标签可以来自各种来源，例如，端点将从本地容器运行时获取与容器相关联的标签，以及从Kubernates获取与Pod相关联的标签，由于这两个标签命名空间彼此不知晓，可能会导致标签冲突，，为了解决潜在的冲突，Cilium在导入标签时，会在所有标签键前加上source:前缀，以标识标签的来源，例如，k8s:role=fronted、container:user=joe、k8s:role=backend，当你使用docker run [...] -l foo=bar运行daocker容器时，Cilium端点将显示标签container:foo=bar。类似地，带有标签foo:bar启动的Kubernates Pod将与标签k8s:foo= bar关联。每个潜在的来源都分配了一个唯一的名称。目前支持以下来源标签：container: 用于从本地容器运行时派生的标签；k8s：用于从Kubernates派生的标签；reserved：用于特殊保留标签，参见特殊标识；unpsec：用于来源未指定的标签。在使用标签识别其他资源时，可以包含来源以限制匹配的特定类型。如果未提供来源，标签来源默认为any:，这将匹配所有来源。如果提供了来源，则选择和匹配的来源需要保持一致。
- 端点(Endpoint)：Cilium通过分配IP地址是容器在网络上可用。多个容器可以共享同一个IP地址，例如，Kubernates Pod，所用共享同一IP地址的容器被称为端点(Endpoint)。
- Identification："Identification" 主要是指如何识别和管理集群节点上的端点。识别机制：端点ID，Cilium为集群节点上的每个端点分配一个内部端点ID，这个ID在单个集群节点上文中是唯一的，用于识别和管理端点。端点ID的唯一性确保了同一个节点上的不同端点可以被明确区分和管理。这对于实现网络策略、负载均衡等至关重要。在Kubernates环境中，一个Pod可能包含多个容器，这些容器共享同一个网络命名空间，Cilium通过分配唯一的端点ID，可以精确的控制和监控网络流量。通过使用唯一的端点ID，Cilium可以实现更细粒度的网络策略控制，提高网络的安全性和可管理性。
- 端点元数据(Endpoint Metadata)：在Cilium中，端点元数据是指与端点相关联的附加信息，这些信息用于识别和管理端点。以便实现安全策略、负载均衡和路由等功能。端点元数据的来源取决于所使用的编排系统和容器运行时。例如，在Kubernates环境中，元数据可以来资源Kubernates Pod标签，而在使用docker的环境中，元数据可以来自于容器标签。元数据用于识别端点，以便在网络策略、负载均衡和路由等操作中使用。通过这些元数据，Cilium可以实现更细粒度的网络控制。元数据以标签的形式附加到端点上，例如，一个容器可能带有标签 app=benchmark，这个标签会与端点关联，并以 container:app=benchmark 的形式表示，表明该标签来自容器运行时。一个端点可以与来自多个来源的元数据相关联。例如，在使用 containerd 作为容器运行时的 Kubernetes 集群中，端点可能会同时具有来自 Kubernetes 的标签（前缀为 k8s:）和来自 containerd 的标签（前缀为 container:）。通过使用元数据，Cilium可以更精确地控制和监控网络流量，从而提高网络的安全性和可管理性。
- 身份(Identity)：在Cilium中，身份(Identity)是一个关键概念，用于管理和强制执行网络策略。身份(Identity)是指分配给每个端点的唯一标识符，用于在端点之间强制执行基本的连接性，这相当于传统网络中的第3层（网络层）强制执行。身份通过标签来识别，每个端点的身份是基于与其关联的Pod或容器的标签派生出来的。这些标签被称为安全相关标签，身份在整个集群范围内是唯一的，所有共享相同安全相关的标签集的端点将共享相同的身份，这种设计使得策略执行可以扩展到大量的端点，因为许多端点通常会共享相同的安全标签集。当Pod或容器启动时，Cilium会根据容器运行时接收到事件创建一个端点，并解析其身份。如果Pod或容器的标签发生变化，Cilium会重新确认并自动更新端点的身份。身份用于实现网络策略、负载均衡和路由等功能。通过使用身份，Cilium可以精确地控制和监控网络流量。
- 安全相关标签(Security Relevant Labels)：在 Cilium 中，安全相关标签（Security Relevant Labels）是用于确定端点身份的关键标签。安全相关标签是指派生端点身份时需要考虑有意义的标签。并非所有与容器或Pod关联的标签都是安全相关的，例如，一些标签可能仅用于存储元数据，如容器启动的时间戳。这些标签用于确定端点的身份，从而在网络策略、负载均衡和路由等操作中使用，通过使用安全相关标签，Cilium可以精确的控制和监控网络流量。为了识别那些标签是安全相关的，用户需要指定一组有意义标签的字符串前缀，所有以前缀id:开头的标签，例如 id.service1、id.service2、id.groupA.service44。可以在启动Cilium代理时指定有意义标签前缀的列表，以便 Cilium 知道哪些标签需要在派生身份时考虑。通过使用安全相关标签，Cilium能够实现更细粒度的网络策略控制，提高网络的安全性和可管理性。
- 特殊身份(Special Identities)：所有由Cilium管理的端点都会被分配一个身份，为了允许与不由Cilium管理的网络端点进行通信，存在特殊身份来表示这些端点，特殊保留身份以字符串reserved: 为前缀。
- 已知身份：以下是 Cilium 自动识别的已知身份列表，这些身份无需联系任何外部依赖（如 kvstore）即可分配安全身份。这样做的目的是允许 Cilium 启动并在集群中为基本服务启用带有策略执行的网络连接，而不依赖于任何外部。
-集群中的身份管理：在 Cilium 中，集群中的身份管理（Identity Management in the Cluster）是确保所有集群节点上的端点能够一致地解析和共享身份的机制。身份在整个集群中都有效，如果在不同集群节点上启动了多个Pod或容器，只要他们共享相同的身份相关标签，他们都将解析并共享一个单一的身份。为了实现这种一致性，集群节点之间需要协调，这通过分布式键值存储来实现，该存储允许执行原子操作，以生成新的唯一标识符。解析端点身份的操作是通过查询分布式键值存储来完成的，每个集群节点创建身份相关的标签子集，然后查询键值存储以派生身份。如果标签集之前没有被查询过，将创建一个新的身份，如果之前已经查询过，则返回初始查询的身份。这种机制确保了集群所有节点对身份的一致理解，从而实现统一的网络策略执行和管理。
- 节点(Node)：Cilium将节点定义为集群中的一个独立成员每个节点必须运行cilium-agent，并且主要以自主方式运行。为了简化与扩展，Cilium代理之间的状态同步尽量减少，仅通过键值存储或数据包元数据。这种设计有助于提高系统的可扩展性和简化管理。
- 节点地址(Node Address)：在 Cilium 中，节点地址(Node Address)是指集群中每个节点的网络地址。Cilium会自动检测节点的 IPv4 和 IPv6 地址。这些地址用于在集群中唯一标识每个节点。当cilium-agent启动时，检测到的节点地址会被打印出来。这有助于管理员在配置和调试过程中快速获取节点的网络信息。节点地址用于在集群中用于在节点间的通信和状态同步。它们是实现网络策略和服务发现的基础。准确的节点地址对预计群的正常运行至关重要。因为它们影响到网络流量的路由和策略的执行。

路由：
在Cilium中，路由(Routing)是指网络中确定数据包从源到目标的路径的过程。它是网络通信的基础，确保数据包能够正确地到达目标节点。Cilium支持多种路由模式，包括封装模式和直接路由模式。封装模式采用隧道技术（如VXLAN或Geneve）在节点之间传输数据包，而直接路由模式则依赖于底层网络基础设施。
- 封装模式(Encapsulation)：是一种用于在集群节点之间传输网络流量的技术。当没有提供特定配置时，Cilium会自动运行在封装模式下。这种模式对底层网络基础设施的要求最少，因此是默认选择。在封装模式下，所有集群节点通过基于UDP的封装协议（如VXLAN或Geneve）形成一组隧道，这些隧道用于在节点之间传输封装的网络流量。所有Cilium节点之间的流量都被封装。原始的网络数据包被包裹在另一个数据包内，以便在隧道中传输。封装模式允许Cilium在不需要对现有网络基础设施进行重大更改的情况下实现网络策略和服务发现。他提供了灵活性和扩展性，适用于各种网络环境。封装模式常用于需要跨越不同网络段或数据中心的集群环境，因为它可以绕过底层的各种网络环境。优势：连接集群节点的网络无需了解 PodCIDR。集群节点可以生成多个路由或链路层域。只要集群节点能够通过 IP/UDP 相互访问，底层网络的拓扑结构无关紧要。由于不依赖于任何底层网络限制，可用的地址空间可能会大得多，并且可以根据 PodCIDR 大小的配置在每个节点上运行任意数量的 pod。与 Kubernetes 等编排系统一起运行时，集群中所有节点的列表（包括它们的关联分配前缀节点）会自动提供给每个代理。新节点加入集群时，会自动纳入网格。封装协议允许携带元数据与网络数据包一起传输。Cilium 利用这一能力来传输元数据，例如源安全身份。身份传输是一种优化，旨在避免在远程节点上进行一次身份查找。缺点：**MTU 开销**，由于添加了封装头，用于有效载荷的最大传输单元（MTU）比原生路由要低（VXLAN 每个网络数据包增加 50 字节）。这导致特定网络连接的最大吞吐量降低。通过启用巨型帧（每 1500 字节有 50 字节的开销 vs 每 9000 字节有 50 字节的开销），可以大大缓解这一问题。
- 本地路由(Native-Routing)：在 Cilium 中，本地路由（Native-Routing）是一种利用底层网络路由能力的数据包转发模式。本地路由是一种数据包转发模式，通过routing-mode: native 启用。他利用Cilium运行的网络路由能力，而不是执行封装。在本地路由模式下，Cilium将所有不是发送到另一个本地端点的数据包委托给Linux内核的的路由子系统。数据包将被路由，就像本地进程发出的数据包一样。连接集群节点的网络必须能够路由PodCIDR。这是因为数据包将通过底层网络进行路由，而不是通过隧道。当配置本地路由时，Cilium会自动在Linux内核中启用IP转发，以确保数据包能够正确路由。本地路由模式减少封装带来的开销，提高了网络性能和效率。它适用于底层网络支持路由PodCIDR的环境，提供了更简单的网络配置。
- AWS ENI：是一种用于在 AWS 环境中实现高性能网络连接的技术。AWS ENI 是一种虚拟网络接口，允许在 AWS 环境中实现高性能的网络连接。它可以直接附加到实例，提供更高的网络吞吐量和更低的延迟。在 Cilium 中，AWS ENI 模式允许每个 Pod 直接使用一个 ENI，从而绕过传统的网络虚拟化层。这种模式下，数据包不需要经过主机网络命名空间，直接通过 ENI 进行传输。通过直接使用 ENI，可以显著提高网络吞吐量和降低延迟，适用于对网络性能要求高的应用场景。减少了网络虚拟化层的复杂性，简化了网络配置和管理。每个 Pod 使用独立的 ENI，提供了更好的网络隔离和安全性。AWS ENI 模式适用于需要高网络性能和低延迟的应用，例如大数据处理、实时数据分析和高性能计算等。使用 AWS ENI 模式需要确保底层 AWS 环境支持 ENI 的创建和管理，并且需要配置相应的 IAM 权限。

IP Address Management (IPAM) 是一种管理和组织IP地址空间的技术和流程。它的主要目标是确保网络中的IP地址分配高效、准确，并且能够支持网络的扩展和变化。IPAM系统可以自动化IP地址的分配和回收，确保IP地址资源的有效利用。支持静态和动态的IP地址分配并能够处理DHCP（动态主机配置协议）服务器。IPAM工具能够跟踪哪些IP地址已被分配，哪些是空闲的，以及每个IP地址的使用情况。提供IP地址的历史记录，帮助网络管理员进行审计和问题排查。IPAM系统可以管理和优化子网划分，确保网络拓扑结构合理，支持VLSM（可变长子网掩码）和CIDR（无类别域间路由），以提高IP地址的利用率。IPAM通常与DNS和DHCP集成，以确保DNS、主机名和DNS记录的一致性，自动更新DNS记录，减少人为错误。提供详细的报告和分析功能，帮助管理员了解IP地址的使用情况和网络健康状况。支持自定义报告，满足不同的管理需求。IPAM可以帮助识别和防止未经授权的IP地址使用增强网络安全性。支持访问控制和审计日志，确保只有授权用户才能管理IP地址。支持与其他网络管理工具和系统的集成，实现自动化的IP地址管理。可以通过API与其他系统进行交互，支持自动化工作流。IPAM负责分配和管理由Cilium管理的网络端点（包括容器等）使用的IP地址。支持多种 IPAM 模式以满足不同用户的需求：
- Kubernetes Cluster Scope：是指在在Kubernates集群中，资源和配置的管理范围是整个集群，而不是单个命名空间或节点。Cluster Scope资源在整个集群中是全局可见和可管理的。常见的Cluster Scope资源包括节点(Nodes)、持久卷(Persistent Volumes)、集群角色(Cluster Roles)和集群角色绑定(Cluster Role Bindings)等，这些资源的管理和配置影响整个集群的行为和功能。由于Cluster Scope资源影响整个集群，因此对这些的资源的访问和修改需要更高的权限，集群管理员需要小心管理这些资源的访问权限，以确保集群的安全性和稳定性。Cluster Scope资源确保集群中的配置一致性，例如，集群范围的网络策略可以确保所有命名空间都遵循相同的网络安全规则。通过在集群范围内管理资源，可以实现更高的可用性和容错能力。例如，持久卷可以在集群中的不同节点之间迁移，以确保数据的高可用性。可以再集群范围内定义和实施策略，例如资源配额、网络策略和安全策略，以确保整个集群的资源使用和安全性符合预期。集群范围的 IPAM 模式为每个节点分配 PodCIDR，并使用每个节点上的主机范围分配器分配 IP 地址。因此，它类似于 Kubernetes 的主机范围模式。不同之处在于，Cilium 操作符将通过 v2.CiliumNode 资源管理每个节点的 PodCIDR，而不是通过 Kubernetes v1.Node 资源由 Kubernetes 分配每个节点的 PodCIDR。这种模式的优势在于，它不依赖于 Kubernetes 配置来分发每个节点的 PodCIDR。
- Kubernetes Host Scope：是一种IP地址管理模式（IPAM），它在每个节点上分配和管理Pod的IP地址。在Host Scope模式下，Kubernates为每个节点分配一个唯一的PodCIDR（POS子网范围）。这个PodCIDR定义了该节点上可以分配给Pod的IP地址范围。每个节点上都有一个本地IP分配器，负责从分配给该节点的PodCIDR中分配IP地址给Pod，IP地址的分配是在节点级别进行的。每个节点独立管理其Pod的IP地址的分配。这种模式通常与Kubernates的网络插件或CNI（容器网络接口）插件集成，以确保Pod之间的网络连接和通信。由于IP地址分配是在节点级别进行的，因此可以减少集群级别的管理开销，每个节点可以根据其资源和需求独立管理IP地址，提供更大的灵活性。需要确保每个节点的 PodCIDR 不重叠，并且需要在节点加入或离开集群时进行适当的管理。这种模式依赖于 Kubernetes 配置为分发每个节点的 PodCIDR，如果 Kubernetes 没有正确配置，可能会导致 IP 地址分配失败。通过使用 Host Scope 模式，Kubernetes 集群可以更高效地管理 Pod 的 IP 地址分配，特别是在大规模集群中。然而，这也需要确保每个节点的 PodCIDR 配置正确，以避免 IP 地址冲突和网络问题。
- Multi-Pool：是一种 IP 地址管理（IPAM）模式，语序在Kubernates集群中使用多个IP地址池来分配Pod的IP地址。这种模式通常用于满足特定的网络需求或优化 IP 地址的使用。Multi-Pool 模式允许定义多个IP地址池，每个池可以有多个IP地址范围。这些池可以用于不同的命名空间、节点、工作负载。通过使用多个IP地址池，可以更灵活的分配IP地址。例如，可以为不同的应用或服务分配不同的IP地址池，以满足特定的网络需求。不同的IP地址池可以用于隔离不同的工作负载或命名空间，增强网络安全性，例如，可以为不同的租户或团队分配不同的IP地址池。通过定义多个IP地址池，可以更高效的使用IP地址资源。例如可以为不同的节点分配不同的IP地址池，以避免IP地址冲突。Multi-Pool 模式可以与Kubernates的网络策略集成，已实现更细粒度的网络访问控制，例如，可以为不同的IP地址池定义不同的网络策略。需要在Kubernates集群中进行适当的配置以启用和管理Multi-Pool模式。这可能包括了IP地址池、配置网络插件和更新集群配置。通过使用 Multi-Pool 模式，Kubernetes 集群可以更灵活和高效地管理 IP 地址分配，满足不同的网络需求和安全要求。然而，由于其 Beta 状态，使用时需要注意其稳定性和兼容性。

Cilium容器网络控制流程(Cilium Container Networking Control Flow)：
- 初始化和配置：在Kubernates集群中部署Cilium时，Cilium代理会在每个节点上运行，Cilium负责管理和配置节点上的网络策略和连接。
- Pod创建：当一个新的Pod被创建时，Kubernates会通知Cilium代理，Cilium代理会为该Pod分配一个IP地址，并配置相关的网络接口。
- 网络策略应用：Cilium使用基于BPF（Berkeley Packet Filter）的技术来实现高效的网络策略。这些策略定义了Pod之间的网络流量规则，例如允许或拒绝特定的网络流量。网络策略会被编译成BPF程序，并加载到内核中执行，以实现高性能的流量控制。
- 数据包处理：当一个Pod发送或接收一个数据包时，Cilium会通过BPF程序对数据包进行处理。BPF程序会根据预定义的网络策略决定是否允许数据包通过。Cilium换支持服务发现或负载均衡，确保数据包能够正确路由到目标Pod。
- 安全组和身份管理：Cilium基于身份的安全模式，允许根据Pod的标签和命名空间来定义网络策略。这使得网络策略将可以集成 Prometheus 和其他监控工具，以实现实时的网络监控和告警。更加灵活和细粒度。Cilium 还支持加密通信，确保数据在传输过程中的安全性。
- 监控日志：Cilium 提供了丰富的监控和日志功能，可以帮助管理员了解网络流量和策略执行情况。

伪装(Masquerading)是一种网络地址转换(NAT)技术，通常用于在私有网络和公共网络之间进行通信。它的主要目的是允许使用 私有IP地址的设备能够与公共网络（如互联网）进行通信。私有IP地址是从特定的地址范围（如RFC1918定义的地址块）中分配的，这些地址在公共网络中不可路由，私有IP地址通常用于局域网（LAN）中的设备，也节省公共IP地址资源。当私有网络中的设备发送数据包到公共网络时，Masquerading会将数据包的源IP地址转化为一个公共IP地址，这个公共IP地址通常是路由器或防火墙的外部接口地址，它在公共网络中是可路由的。当公共网络中的设备回复数据包时，Masquerading会将目的IP地址转换回原始的私有IP地址，以确保数据包能正确路由到私有网络中的设备。为了跟踪和区分来自不同私有IP地址的流量，Masquerading通常会使用端口映射技术，每个出站连接都会被分配一个唯一的端口号，以便在返回流量时能够正确映射回原始的私有IP地址和端口。Masquerading提供了一定的安全性。因为私有网络中的设备不会直接暴露在公共网络中，只有经过转换的公共IP地址可见，私有IP地址被隐藏了起来。Masquerading 广泛应用于家庭网络、企业内部网络和数据中心等场景，以实现私有网络与公共网络的互联。在 Kubernetes 等容器编排平台中，Masquerading 用于允许使用私有 IP 地址的 Pod 与外部网络通信。通过使用Masquerading，私有网络中的设备可以安全地与公共网络进行通信，同时节省了公共IP地址资源。

eBPF-Masquerading作为现代云原生网络架构的核心技术之一，通过将传统的iptables的NAT功能迁移至eBPF虚拟机执行，实现了网络性能的突破性提升。研究表明，该技术在A800 GPU集群中可实现3.15倍的吞吐量提升，同时将网路延迟降低至亚毫秒级，为大规模容器化部署提供了新的技术范式。传统Linux网络栈依赖iptables实现SNAT(Source Network Address Translation)，其链式规则匹配机制导致时间复杂度达到O(n^2)级别。在Kubernates集群中，当Pod数量超过5000时，iptables的规则条目可能突破20万条，导致数据包处理延迟急剧上升至50ms以上。更严重的是，iptables的全局锁机制使得并发规则更新成为性能瓶颈，每秒仅能处理200次规则变更请求。eBPF通过在内核空间引入沙盒化虚拟机，将网络处理逻辑编译为字节码直接注入数据路径。Cilium的eBPF-Masquerading实现采用BPF_PROG_TYPE_CGROUP_SKB程序类型，在内和网络协议栈的egress节点插入处理逻辑，当数据包通过虚拟以太网设备(veth)离开Pod时，eBPF程序通过bpf_skb_store_bytes函数直接修改IP头部的原地址字段，将其替换为宿主机的出口IP。该过程完全绕过了iptables，将地址转换操作的时间复杂度将至O(1)。关键数据结构struct bpf_msgq封装了NAT映射信息，包含原始源IP(pod_ip)、转换后IP(node_ip)及会话标识符。通过BPF_MAP_TYPE_LRU_HASH类型映射表实现连接跟踪，其哈希碰撞率控制在0.3%以下，显著优于传统conntrack的链表结构。实验数据显示，单核CPU可以处理200万次/秒的NAT操作，相比iptables提升达8倍。

IPv4分片处理是网络协议栈中解决数据包超过链路MTU限制的核心机制，其实现涉及复杂的状态管理与性能优化。结合传统网络栈与Cilium的eBPF创新方案，如下：
- 分片机制管理：当IPv4数据包大小超过路径MTU时，路由器与主机执行分片操作。分片规则，每个分片携带原始IP头并修改总长度、片偏移和MF标志。例如：4000字节数据包（3980字节有效负载）在1500字节MTU链路分片：分片1:1480B数据，偏移0，MF= 1；分片2:1480B数据，偏移185（1480/8）MF=1；分片3:1020B数据，偏移370(2960/8)MF=0。所有分片保持相同的标识符字段以实现重组。接收端需要缓存所有分片直至最后一个到达，超时机制（30秒）防止资源耗尽，内存攻击风险：恶意发送大量不完整分片耗尽系统资源。
- 优化：Cilium通过eBPF重构分片处理流程：
```c
// eBPF分片跟踪核心逻辑
struct bpf_map_def SEC("maps") ipv4_frag_map = {
    .type = BPF_MAP_TYPE_LRU_HASH,
    .key_size = sizeof(struct ipv4_frag_key),
    .value_size = sizeof(struct ipv4_frag_value),
    .max_entries = 1024 * 1024,
};

SEC("xdp")
int handle_frag(struct xdp_md *ctx) {
    struct iphdr *iph = data_ptr(ctx);
    if (iph->frag_off & IP_MF || iph->frag_off & IP_OFFSET) {
        struct ipv4_frag_key key = { .saddr = iph->saddr, .id = iph->id };
        struct ipv4_frag_value *frag = bpf_map_lookup_elem(&ipv4_frag_map, &key);
        // 分片状态跟踪与重组逻辑
    }
    return XDP_PASS;
}
```
连接跟踪表项从链表改为了LRU哈希，查询复杂度O(1) -> O(1)，分片缓存内存降低75%（256MB -> 64MB/节点）。该技术演进使得Kubernates集群在万节点规模下实现99% 延迟小于1.2ms，同时将NAT吞吐量提升至120Gbps/节点。未来随着智能网卡对eBPF的硬件卸载支持，分片处理性能有望突破400Gbps。

Kubernates Network：

在Kubernates集群中运行Cilium时，提供以下功能：CNI插件支持，为Pod提供联网功能，并支持多集群网络。基于身份的NetworkPolicy实现，隔离三层和四层网络中的Pod to Pod连接。NetworkPolicy的CRD扩展，通过自定义资源定义（CRD）扩展网络策略控制，包括：七层策略执行，在入站和出站流量中，对HTTP、Kafka等应用协议进行七层策略执行。CIDR出站支持，保护对外部服务的访问。外部无头服务强制限制，自动将外部无头服务限制为服务配置的Kubernates端点集。ClusterIP实现，为Pod toPod流量提供分布式负载均衡。与现有kube-proxy模型完全兼容。这些功能使Cilium能够提供更细粒度的网络控制和安全性，同时支持多集群环境下的网络通信和策略执行。

Pod间连接：在Kubernates集群中，容器部署在Pod内，每个Pod包含一个或多个容器，并通过一个单一的IP地址进行访问，使用Cilium时，每个Pod从运行该Pod的Linux节点的前缀中获得一个IP地址。在没有任何网络安全策略的情况下，所有Pod都可以相互访问。Pod的IP地址通常局限于Kubernates集群内部。如果Pod需要作为客户端访问集群外部的服务，则当网络流量离开节点时，会自动进行伪装。每个Pod都被分配一个唯一的IP地址，这使得Pod可以像虚拟机或物理主机一样进行通信。无NAT通信，Kubernates允许在所有Pod之间直接通信，无需网络地址转换（NAT）。Kubernates网络模型抽象了底层网络基础设施，使用CNI(Container Network Interface)插件来实现Pod之间的通信。CNI插件负责在Kubernates集群中建立网络连接。常见的CNI插件包括：Layer2（以太网）解决方案，使用ARP和以太网交换来实现Pod之间的通信。Layer3（路由）解决方案，通过路由来连接不同节点上的Pod。Overlay网络：在现有网络基础设施上建立虚拟网络。已实现Pod之间的通信。Cilium是基于eBPF的网络解决方案，它为每个Pod分配一个IP地址，并使用额BPF程序来管理网络流量。Cilium支持基于身份的网络策略，允许对Pod之间的通信进行细粒度的控制。Kubernates提供了NetworkPolicy资源来控制Pod之间的流量。Cilium扩展了这一功能，支持基于身份的网络策略和七层网络策略。Cilium基于身份的网络策略来确保Pod之间的安全通信，减少了网络攻击的风险。Pod间通信：同一节点上的Pod可以直接用localhost进行通信。不同节点的Pod可以通过其分配的IP地址直接通信，无需NAT。Pod-to-Pod连接是Kubernetes网络模型的关键组成部分，通过CNI插件和基于eBPF的解决方案来实现高效、安全的Pod间通信。

服务负载均衡(Service Load-balancing)：

Kubernates提供了Services抽象，允许用户在不同的Pod之间负载均衡网络流量。这种抽象使得Pod可以通过一个单一的IP地址（虚拟IP地址）访问其它Pod，而无需知道运行该服务的所有Pod。在没有Cilium的情况下，Kube-proxy会安装在每个节点上，建设kube-master上的端点和服务的添加和删除，从而能够在iptables上应用必要的规则。因此，从Pod发送和接收的流量会被正确的路由到为该服务提供服务的节点和端口。在实现了ClusterIP时，Cilium遵循与kube-proxy相同的原则，监视服务的添加和删除，但不同的是，它不是在iptables上执行规则，而是更新每个节点上的eBPF映射条目。服务类型：
- ClusterIP：是Kubernates中常见服务类型，它为服务内部分配一个集群内部可访问的虚拟IP地址。kube-proxy：在每个节点上运行，监视服务和端点的添加和删除，并在iptables上应用必要的规则，以确保流量正确路由到提供服务的节点和端口。Cilium：与kube-proxy类似，但它通过更新每个节点上的eBPF映射条目来实现流量路由，而不是修改iptables。
- NodePort：此类型的服务在每个节点上开放一个特定端口，允许外部流量通过节点IP和端口访问服务。仅在节点可达时才有效，通常用于私有网络环境。
- LoadBalancer：此类型的服务在云提供商的基础设施中自动创建一个负载均衡器，提供外部网络访问。通过将流量分配给多个Pod，确保服务的高可用性和可扩展性。

负载均衡器(LoadBalancer)：将入站流量分配到多个Pod，确保没有单个Pod过载，从而提高性能和可用性。服务使用一个虚拟IP地址，使得客户端无需知道后端Pod的具体地址。通过eBPF实现流量路由，相比iptables具有更高的性能和灵活性。使用云提供商自动创建负载均衡器，配置健康检查和自动扩缩以优化服务性能。

部署：标准Cilium Kubernates部署的配置包括以下几种Kubernates资源：
- DaemonSet：描述了部署到每个Kubernates节点上的Cilium Pod，这个Pod运行cilium-agent及其相关的守护进程。该DaemonSet的配置包括镜像标签，指示Cilium Docker容器的确切版本（例如，v1.0.0），以及传递给cilium-agent的命令行选项。
- ConfigMap：描述了传递给cilium-agent的常见配置，例如kvstore端点和凭据、启用或禁用调试模式。
- ServiceAccount、ClusterRole和ClusterRoleBindings：这些资源定义了cilium-agent访问Kubernates API服务器所使用的身份和权限，前提是启用了Kubernates RBAC。

在Cilium DaemonSet部署之前已经运行的Pod将继续使用之前的网络插件进行连接，具体取决于CNI配置。一个典型的例子是kube-dns服务，它默认在kube-system命名空间中运行。更改现有Pod的网络连接的一种简单方法是利用Kubernates的特性，即如果Pod被删除，Kubernates会自动重新启动Deployment中Pod。因此，可以删除原来的kube-dns Pod，随后立即启动被替换的Pod，并由Cilium管理网络连接。在生产环境中，可以通过对kube-dns Pod滚动更新来执行，以避免DNS服务中断。kubectl --namespace kube-system get pods 可以查看kube-dns集合状态列表。Kubernates可以通过活性探针(Liveness Probes)和就绪探针(Readiness Probes)来标识应用程序的健康状态，为了使kubelet能够在每个Pod上运行健康检查，默认情况下，Cilium将始终允许来自本地主机的所有入站流量进入到每个Pod。

网络策略(Network Policy)：在Kubernates上运行Cilium时，可以利用Kubernates的分发策略，有三种方式可以用来配置Kubernates的网络策略：
- 标准NetworkPolicy资源：支持在Pod的入口和出口处配置L3和L4策略。
- 扩展的CiliumNetworkPolicy：作为自定义资源定义(CustomResourceDefinition)支持L3到L7为入口和出口配置策略。
- CiliumClusterwideNetworkPolicy：这是一个集群范围内的自定义资源定义，用于指定由Cilium强制执行的集群范围策略。其规范与 CiliumNetworkPolicy 相同，但没有指定命名空间。

CiliumNetworkPolicy是Cilium提供的一种网络策略资源，用于在Kubernates集群中提供更细粒度的安全控制，它扩展了标准的Kubernates NetworkPolicy，支持OSI模型的第三层（L3）、第四层（L4）和第七层（L7）定义网络访问规则。允许用户基于标签、IP地址、DNS名称等条件定义网络访问规则，实现精确控制哪些Pod可以相互通信，以及可以使用哪些协议和端口。支持L3和L4的网络策略，同时在L7层提供常见协议（如HTTP、gRPC、Kafka）的支持。除了命名空间范围的策略，Cilium还提供CiliumClusterwideNetworkPolicy，用于在整个集群中强制强制实施一致的安全策略。将安全性与工作负载解耦，利用标签和元数据来管理网络策略，从而避免了因IP地址变化而频繁更新安全规则的问题。CiliumNetworkPolicy的结构：Metadata：描述策略的元数据，包括策略名称、命名空间和标签。Spec：包含一个规则基础的字段，用于定义具体的网络策略规则。Specs：包含规则基础列表的字段，适用于余姚自动添加或移除多个规则的情况。Status：提供是否成功应用了策略的状态。CiliumNetworkPolicy适用于需要细粒度网络策略的微服务架构，特别是Kubernates环境中，它可以帮助用户在应用层面定义更复杂的访问控制规则。提高集群的安全性和可观测性。
```go
type CiliumNetworkPolicy struct {
        // +deepequal-gen=false
        metav1.TypeMeta `json:",inline"`
        // +deepequal-gen=false
        metav1.ObjectMeta `json:"metadata"`

        // Spec is the desired Cilium specific rule specification.
        Spec *api.Rule `json:"spec,omitempty"`

        // Specs is a list of desired Cilium specific rule specification.
        Specs api.Rules `json:"specs,omitempty"`

        // Status is the status of the Cilium policy rule
        // +deepequal-gen=false
        // +kubebuilder:validation:Optional
        Status CiliumNetworkPolicyStatus `json:"status"`
}
```

CiliumClusterwideNetworkPolicy(CCNP) 与 CiliumNetworkPolicy 类似但有以下两个区别：非命名空间和集群范围，有CiliumClusterwideNetworkPolicy定义的策略是非命名空间的，并且适用于整个集群范围。启用节点选择器，它允许使用节点选择器来定义策略。CiliumClusterwideNetworkPolicy(CCNP)是Cilium提供的集群级网络策略资源，用于在Kubernates集群中实施全局安全规则。与标准的NetworkPolicy和CiliumNetworkPolicy不同，CCNP具备跨命名空间控制能力，可覆盖整个集群的网络流量。策略规则自动应用于所有命名空间，无需为每个命名空间单独配置，支持节点级别的网络访问控制，例如kubelet API的访问范围。作为最高优先级策略层，可覆盖应用级别的网络策略冲突。与CiliumNetworkPolicy形成层次化的策略体系，实现默认拒绝 + 例外放行的零信任模型。CCNP与CiliumNetworkPolicy配合使用，CCNP定义基础安全边界，应用级策略处理业务逻辑。通过Hubble监控策略的执行效果，确保策略规则未阻断正常业务流量。在混合云场景中结合ClusterMesh实现跨集群策略同步。
```go
type CiliumClusterwideNetworkPolicy struct {
        // Spec is the desired Cilium specific rule specification.
        Spec *api.Rule

        // Specs is a list of desired Cilium specific rule specification.
        Specs api.Rules

        // Status is the status of the Cilium policy rule.
        //
        // The reason this field exists in this structure is due a bug in the k8s
        // code-generator that doesn't create a `UpdateStatus` method because the
        // field does not exist in the structure.
        //
        // +kubebuilder:validation:Optional
        Status CiliumNetworkPolicyStatus
}
```

CiliumCIDRGroup(CCG)：是Cilium提供的一种用于管理CIDR块的功能，允许管理员在CiliumNetworkPolicy中引用一组CIDR块，这种机制使得网络策略的配置更加灵活和高效，尤其适用于对外部CIDR块进行策略控制的场景，CiliumCIDRGroup目前处于Beta阶段，与Cilium Agent管理的Endpoint资源不同，CiliumCIDRGroup需要管理员手动管理。任何与CiliumCIDRGroup相关的流量都会被注释为CCG的名称和标签，这有助于流量的监控和分析。以下是一个CiliumCIDRGroup的示例配置：
```yaml
apiVersion: cilium.io/v2alpha1
kind: CiliumCIDRGroup
metadata:
  name: vpn-example-1
  labels:
    role: vpn
spec:
  externalCIDRs:
  - "10.48.0.0/24"
  - "10.16.0.0/24"

# 然后，可以在CiliumNetworkPolicy中通过fromCIDRSet或toCIDRSet指令引用这个CIDR组：

apiVersion: cilium.io/v2
kind: CiliumNetworkPolicy
metadata:
  name: from-vpn-example
spec:
  endpointSelector: {}
  ingress:
  - fromCIDRSet:
    - cidrGroupRef: vpn-example-1
```

CRD(Custom Resource Definition)是Kubernates中用于扩展API的机制，允许用户定义新的资源类型。通过CRD，可以再集群中注册自定义资源，并使用Kubernates的原生API和工具进行管理。在Kubernates中管理Pod时，Cilium会为每个由Cilium管理的Pod创建一个自定义资源定义(CRD)，类型为CiliumEndpoint。每个CiliumEndpoint的名称和命名空间与对应的Pod相同。CiliumEndpoint对象包含与 cilium-dbg endpoint get 命令的JSON输出相同的信息，位于.status字段下，并且可以为集群中所有Pod获取这些信息。通过添加 -o json 选项，可以导出有关每个端点的更多信息。这包括端点的标签、安全身份以及对其生效的策略。

##### Kubernates Network

Kubernates网络，Kubernates网络模型：
- 每个Pod都有自己的IP地址。
- Pod内的容器共享Pod的IP地址，并且可以相互自由通信。
- Pod可以使用自己的IP地址与集群中其他所有的Pod进行通信（无需NAT）。
- 使用网络策略来定义隔离（限制每个Pod可以通信的内容等）。

Pod可以被看作主机或虚拟机一样（它们都拥有唯一的 IP 地址），Pod中的容器则非常像主机或虚拟机中运行的进程（它们运行在同一个网络命名空间中共享一个 IP 地址）。这种模型使得微服务更容易从主机或虚拟机迁移到Kubernates管理的Pod中。由于隔离使用**网络策略**，而不是网络结构，因此还是很容易理解的。这种网络类型被称为”**扁平网络**“。当然Kubernates也支持将主机端口映射到Pod的功能，或者共享主机IP地址的主机网络命名空间内直接运行Pod。Kubernates内置的网络支持kubenet，kubenet提供一些基本的网络连接。最常见的做法还是使用第三方插件，通过 CNI（容器网络接口）API 插入 Kubernetes。CNI插件主要两种：网络插件，负责将Pod连接到网络；IPAM（IP地址管理）插件，负责分配Pod IP地址。

Kubernates服务提供了一种对一组Pod的访问抽象为网络服务的方法，这组Pod通常使用标签选择器来定义，在集群内，网络服务通常表示为虚拟IP地址，kube-proxy会在这组支持服务的Pod之间负载均衡连接到虚拟IP。虚拟IP可以通过Kubernates DNS来发现。DNS名称和虚拟IP地址在服务的生命周期内保持不变，即使支持服务的Pod可能被创建或销毁，支持服务的Pod数量也可能随时间变化。Kubernates服务还可以定义从集群外部访问服务的的方式。例如，使用节点端口（NodePort），服务可以通过集群中每个节点上的特定端口访问。或使用负载均衡器，网络负载均衡器提供了一个虚拟IP地址，服务可以通过该地址从集群外部访问。

Kubernates DNS服务，每个Kubernates集群都提供一个DNS服务。每个Pod和每个服务都可以通过Kubernates DNS 服务发现。Kubernates DNS服务是作为Kubernates 服务实现的，该服务映射到一个或多个DNS Pod（通常是 CoreDNS），这些DNS Pod像其他任何Pod一样被调度。集群中的Pod被配置为Kubernates DNS服务，DNS 搜索列表包括 Pod 自身的命名空间和集群的默认域。如果在 Kubernetes 命名空间 bar 中有一个名为 foo（Pod命名空间）的服务，那么同一命名空间中的 Pod 可以通过 foo（Pod命名空间）访问该服务，而其他命名空间中的 Pod 可以通过 foo（集群命名空间）.bar（Pod命名空间） 访问该服务。

出站NAT（Network Address Translation，网络地址转换）：是一种网络技术，用于在数据包从内部网络发送到外部网络是，修改数据包的源IP地址。NAT 的另一个常见用例是负载均衡。在这种情况下，负载均衡器会执行 DNAT（目标网络地址转换），将传入连接的目标 IP 地址更改为其要进行负载均衡的设备的 IP 地址。然后，负载均衡器会对响应数据包进行反向 NAT，这样源设备和目标设备都不会察觉到映射正在进行。
- 源地址转换：当内部网络中的设备发送数据包到外部网络时，NAT设备（如路由器或防火墙），会将数据包的源IP地址从私有IP地址转换为公有IP地址。
- 端口映射：为了区分不用内部设备的数据包，NAT设备会使用端口映射技术，将不同的内部IP地址和端口号映射到公有IP地址和端口号。
- 返回数据包处理：当外部网络返回数据包时，NAT设备会根据端口映射表，将公有IP地址和端口号转换回原始的私有IP地址和端口号，并将数据包转发给相应的内部设备。

应用场景：
- 家庭和办公网络（地址转换）：家庭和办公网络中的路由器通常使用出站 NAT，将内部设备的私有 IP 地址转换为公有 IP 地址，使其能够访问互联网。
- 数据中心和云计算（解决IP地址不足，隐藏内部网络结构）：在数据中心和云计算环境中，出站 NAT 用于隐藏内部网络结构，增强安全性，并解决 IP 地址不足的问题。
- Kubernetes 集群（地址转换）：在 Kubernetes 集群中，出站 NAT 用于将 Pod 的私有 IP 地址转换为节点的公有 IP 地址，使 Pod 能够与外部网络通信。

- 优点：增强安全性：通过隐藏内部网络结构，出站 NAT 可以增强网络安全性，防止外部攻击。解决 IP 地址不足：通过复用公有 IP 地址，出站 NAT 可以解决 IPv4 地址资源有限的问题。简化网络管理：出站 NAT 可以简化网络管理，减少对公有 IP 地址的需求。
- 缺点：性能开销：地址转换和端口映射会带来一定的性能开销，可能影响网络性能。复杂性：出站 NAT 的配置和管理相对复杂，需要专业的网络知识和技能。兼容性问题：某些应用和协议可能不兼容 NAT，导致通信问题。

DNS：虽然网络数据包在网络中的传输是通过 IP 地址确定的，但用户和应用程序通常希望使用众所周知的名称来标识网络目的地，这些名称即使底层 IP 地址发生变化也保持一致。例如，将 google.com 映射到 216.58.210.46。这种从名称到 IP 地址的转换是由 DNS（域名系统）处理的。DNS 运行在迄今为止描述的基础网络之上。每个连接到网络的设备通常配置有一个或多个 DNS 服务器的 IP 地址。当应用程序想要连接到一个域名时，会向 DNS 服务器发送一条 DNS 消息，DNS 服务器随后会响应该域名映射到哪个 IP 地址的信息。然后，应用程序可以根据选定的 IP 地址启动连接。

MTU（最大传输单元）：网络链路的最大传输单元（MTU）是指可以通过该网络链路发送的最大数据包大小。通常，网络中的所有链路都配置相同的 MTU，以减少数据包在穿越网络时需要分片的情况，从而显著提高网络性能。此外，TCP 会尝试学习路径 MTU，并根据网络路径中任何链路的最小 MTU 调整每个网络路径的数据包大小。当应用程序尝试发送超过单个数据包容量的数据时，TCP 会将数据分片为多个 TCP 段，以确保不超过 MTU。大多数网络的链路 MTU 为 1,500 字节，但某些网络支持 9,000 字节的 MTU。在 Linux 系统中，较大的 MTU 可以导致 Linux 网络栈在发送大量数据时使用的 CPU 更少，因为它需要处理的数据包数量更少。

覆盖网络(Overlay Networks)：覆盖网络是一种现有物理网络（称为底层网络，Underlay）基础上构建的虚拟网络技术。它通过在网络层之上添加一层逻辑网络，使得网络流量可以在不同物理网络之间传输，而不需要修改底层网络基础设施。覆盖网络广泛应用于数据中心、云计算和容器编排等场景。特别是在需要跨越多个物理网络或子网的情况下。从连接到覆盖网络的设备的角度来看，它看起来就像一个普通的网络。有许多不同类型的覆盖网络使用不同的协议来实现这一点，但总体来说，它们都具有相同的共性，即将一个网络数据包（称为内部数据包）封装在一个外部网络数据包中。通过这种方式，底层网络只看到外部数据包，而不需要理解如何处理内部数据包。在其上“叠加”一层虚拟网络，实现业务流量的隔离、灵活路由和多租户支持。覆盖网络如何知道将数据包发送到哪里，取决于覆盖网络的类型和它们使用的协议。同样，数据包的具体封装方式在不同类型的覆盖网络之间也有所不同。例如，在 VXLAN 的情况下，内部数据包被封装并作为 UDP 数据包在外部数据包中发送。
- 逻辑网络构建：Overlay网络通过封装技术（如VXLAN、GRE等）将数据包包裹在新的封装中，在底层网络上建立虚拟隧道，实现不同物理位置的节点间二层或三层连接。
- 解耦物理与逻辑：Overlay网络与底层物理网络相互独立，底层负责数据包的传输，Overlay负责网络拓扑和策略的灵活定义。
- 多租户支持：Overlay网络能支持多个虚拟网络共存，即使它们使用相同的IP地址，也不会冲突，适合云计算和数据中心环境。
- 灵活路由与多路径：支持多路径转发和动态路由，提升网络带宽利用率和容错能力。
- 安全隔离：通过加密和虚拟网络划分，保护私有流量的安全传输。

常见协议：
- VXLAN（Virtual Extensible LAN）：基于UDP封装的二层虚拟网络，利用24位VNI标识不同虚拟网络，广泛应用于数据中心。
- NVGRE（Network Virtualization using Generic Routing Encapsulation）：基于GRE协议的隧道技术，支持虚拟网络隔离。
- GRE（Generic Routing Encapsulation）：通用路由封装协议，用于封装各种网络层协议。
- STT（Stateless Transport Tunneling）：利用TCP头优化网卡特性，提高传输效率。

应用场景：
- 云数据中心：通过Overlay网络实现不同租户的网络隔离和灵活部署。
- SD-WAN：利用Overlay网络实现广域网的灵活管理和优化。
- 容器网络：Kubernetes等容器平台常用Overlay网络（如Flannel、Calico的VXLAN模式）实现Pod间跨主机通信。

L3 网络层引入了 IP 地址，通常标志着应用开发人员关心的网络部分和网络工程师关心的网络部分之间的界限。特别是，应用开发人员通常将 IP 地址视为网络流量的源和目的地，但很少需要理解 L3 路由或网络栈中更低层次的内容，这些通常是网络工程师的领域。IP 地址组通常使用 CIDR 表示法表示，该表示法由一个 IP 地址和 IP 地址上的有效位数组成，中间用斜杠分隔。例如，192.168.27.0/24 表示从 192.168.27.0 到 192.168.27.255 的 256 个 IP 地址组。单个 L2 网络中的 IP 地址组称为子网。在子网内，数据包可以在任意两个设备之间作为单个网络跳转发送，仅基于 L2 头（和尾）。要将数据包发送到单个子网之外，需要 L3 路由，每个 L3 网络设备（路由器）负责根据 L3 路由规则决定数据包的路径。每个作为路由器的网络设备都有路由，确定特定 CIDR 的数据包应发送到的下一跳。例如，在 Linux 系统中，路由 10.48.0.128/26 via 10.0.0.12 dev eth0 表示目的 IP 地址在 10.48.0.128/26 的数据包应通过 eth0 接口路由到下一跳 10.0.0.12。路由可以由管理员静态配置，也可以使用路由协议动态编程。使用路由协议时，每个网络设备通常需要配置，以告知它应与哪些其他网络设备交换路由。然后，路由协议会处理在添加或移除设备或网络链路进出服务时，在整个网络中编程正确的路由。BGP 是推动互联网运行的主要协议之一，因此具有极高的可扩展性，并被现代路由器广泛支持。

网络策略(Network Policy)：网络策略是 Kubernetes 网络的主要工具。用于限制集群中的网络流量。历史上，在企业网络中，网络安全是通过设计网络设备（交换机、路由器、防火墙）的物理拓扑结构及其相关配置来实现的。物理拓扑定义了网络的安全边界。在虚拟化的第一阶段，相同的网络和网络设备结构被虚拟化到云中，并使用相同的技术来创建特定的（虚拟）网络设备拓扑结构，以提供网络安全。添加新应用程序或服务通常需要额外的网络设计，以更新网络拓扑和网络设备配置，以提供所需的安全性。相比之下，Kubernetes 网络模型定义了一个“扁平”网络，其中每个 Pod 都可以使用 Pod IP 地址与集群中的所有其他 Pod 通信。这种方法极大地简化了网络设计，并允许新的工作负载在集群中的任何位置动态调度，而不依赖于网络设计。在这种模型中，网络安全不再由网络拓扑边界定义，而是通过与网络拓扑无关的网络策略来定义。网络策略进一步从网络中抽象出来，使用标签选择器作为其主要机制，用于定义哪些工作负载可以与哪些工作负载通信，而不是使用 IP 地址或 IP 地址范围。虽然你可以（也应该）使用防火墙来限制网络边界的流量（通常称为南北向流量），但它们对 Kubernetes 流量的控制能力通常仅限于整个集群的粒度，而不是特定的 Pod 组，这是由于 Pod 调度和 Pod IP 地址的动态特性。此外，大多数攻击者一旦在边界内获得一个小的立足点，其目标就是横向移动（通常称为东西向）以访问更高价值的目标，而基于边界的防火墙无法防范这种行为。另一方面，网络策略是为 Kubernetes 的动态特性而设计的，它遵循标准的 Kubernetes 范式，使用标签选择器来定义 Pod 组，而不是 IP 地址。由于网络策略在集群内部执行，因此它可以同时控制南北向和东西向流量。网络策略代表了网络安全的重要演进，不仅因为它处理了现代微服务的动态特性，还因为它赋予开发和 DevOps 工程师自行定义网络安全的能力，而不需要学习底层网络细节或向负责管理防火墙的单独团队提交工单。网络策略使得定义意图（例如“只有这个微服务可以连接到数据库”）变得容易，将这种意图编写为代码（通常在 YAML 文件中），并将网络策略的编写集成到 Git 工作流和 CI/CD 流程中。

Kubernetes 网络策略是使用 Kubernetes 的 NetworkPolicy 资源定义的。Kubernates的网络策略的特点：
- 策略是命名空间范围的。
- 策略通过标签选择器应用于Pod。
- 策略规则可以指定Pod、命名空间或CIDR的流量。
- 策略规则可以指定协议(TCP、UDP、SCTP)、命名端口或端口号。

Kubernetes 本身不执行网络策略，而是将其执行委托给网络插件（CNI）。网络策略的最佳实践：
- 入站和出站流量(Ingress & egress)：建议每个 Pod 都应该得到网络策略入站规则的保护，这些规则限制连接到 Pod 的流量以及允许的端口。还包括定义网络策略出站规则，这些规则限制 Pod 的出站连接。入站规则保护 Pod 免受来自 Pod 外部的攻击。出站规则有助于在 Pod 被入侵时保护 Pod 外部的所有内容，减少攻击面，防止攻击者横向移动（东西向）或将受损数据从集群中窃取（南北向）。
- 策略模式：由于网络策略和标签的灵活性，通常有多种标签和编写策略的方式来实现这一目标。一种最常见的方法是定义少量适用于所有 Pod 的全局策略，然后为每个 Pod 定义一个特定的策略，该策略定义了所有特定于该 Pod 的入站和出站规则。
```yaml
kind: NetworkPolicy
apiVersion: networking.k8s.io/v1
metadata:
  name: front-end
  namespace: staging
spec:
  podSelector:
    matchLabels:
      app: back-end
  ingress:
    - from:
        - podSelector:
            matchLabels:
              app: front-end
      ports:
        - protocol: TCP
          port: 443
  egress:
    - to:
        - podSelector:
            matchLabels:
              app: database
      ports:
        - protocol: TCP
          port: 27017
```
- 默认拒绝(Default Deny)策略：定义默认拒绝的网络策略。每当部署一个新的 Pod 时，会为该Pod自动定义一个默认拒绝策略。并且可以包含一些对默认拒绝的例外（例如，允许 Pod 访问 DNS）。
```yaml
# 定义一个默认拒绝策略（kube-dns/core-dns的DNS查询除外）

apiVersion: projectcalico.org/v3
kind: GlobalNetworkPolicy
metadata:
  name: default-app-policy
spec:
  namespaceSelector: has(projectcalico.org/name) && projectcalico.org/name not in {"kube-system", "calico-system", "calico-apiserver"}
  types:
    - Ingress
    - Egress
  egress:
    - action: Allow
      protocol: UDP
      destination:
        selector: k8s-app == "kube-dns"
        ports:
          - 53
```
- 分层策略：通过角色来分层定义网络策略。

Kubernetes Ingress：Kubernates Ingress是基于Kubernates Services，提供应用层的负载均衡，将特定域名或URL的HTTP和HTTPS请求映射到Kubernates Services，Ingress还可以用于在负载均衡到服务之前终止SSL/TLS。Ingress的实现细节取决于如何使用Ingress Controller，Ingress Controller负责监控Kubernates Ingress资源，并配置一个或多个Ingress负载均衡器来实现应用的负载均衡，常见的 Ingress Controller 包括 NGINX Ingress Controller、Traefik、HAProxy等。与处理网络层(L3-4)的Kubernates Services不同，Ingress负载均衡器在应用层(L5-7)使用，请求通过负载均衡器与选定的服务后端 Pod 之间的单独连接进行转发。Kubernates Ingress 可以将外部流量分发到集群内的多个服务实例，从而实现负载均衡。Kubernates Ingress 支持基于域名、URL 路径等多种路由规则，可以灵活地将流量路由到不同的服务。Kubernates Ingress 可以在负载均衡之前终止 SSL/TLS 连接，简化证书管理。通过 Kubernates Ingress，可以将多个服务暴露在同一个外部 IP 地址下，简化客户端配置。
- 集群内 Ingress：集群内 Ingress 是指在 Kubernetes 集群内部的 Pod 中执行 Ingress 负载均衡。在这种情况下，负载均衡器作为集群内的一个或多个 Pod 运行，处理进入集群的流量并将其路由到相应的服务。适用于需要高度灵活性和自定义配置的场景，特别是在私有云或本地数据中心环境中。
- 外部 Ingress：外部 Ingress 是指在 Kubernetes 集群外部实现 Ingress 负载均衡。在这种情况下，负载均衡由集群外部的设备或云服务提供商的功能来实现。这些外部设备或服务负责处理进入集群的流量，并将其路由到集群内的相应服务。适用于希望减少集群内资源消耗、利用云服务提供商功能的场景，特别是在公有云环境中。

Kubernates Egress：用来描述Pod到集群外部任何目标的连接，与Kubernetes Ingress不同，Kubernates提供了Ingress资源来管理流量，但没有Kubernates Ingress资源。Kubernates Egress在网络层的处理方式取决于集群使用的Kubernates网络实现（CNI插件）。如果使用服务网格(Service Mesh)，它可以在Kubernates网络实现(CNI插件)的基础上增加出站功能。对于Kubernetes Ingress可以有是那种选择方式：
- 限制出站流量：控制哪些Pod可以连接到集群外部的设备。通过使用网络策略（Network Policy）为每个微服务定义出站规则来实现的，往往与一个默认拒绝策略结合使用，该策略确保在定义允许特定流量的策略之前，默认拒绝所有出站流量。使用 Kubernetes 网络策略限制对特定外部资源的访问时，一个限制是外部资源需要在策略规则中指定为 IP 地址（或 IP 地址范围）。如果与外部资源关联的 IP 地址发生变化，则每个引用这些 IP 地址的策略都需要更新为新的 IP 地址。
- 出站NAT：出站流量如何进行网络地址转换（NAT），如何处理源IP地址。网络地址转换（NAT）是将数据包中的 IP 地址映射到另一个 IP 地址的过程，数据包在通过执行 NAT 的设备时进行这一操作。根据使用场景的不同，NAT 可以应用于源 IP 地址或目标 IP 地址，或者同时应用于两者。如果覆盖网络中的 Pod 尝试连接到集群外部的 IP 地址，则托管该 Pod 的节点会使用 SNAT（源网络地址转换）将数据包的不可路由源 IP 地址映射到节点的 IP 地址，然后转发数据包。节点随后将反向传入的响应数据包映射回原始的 Pod IP 地址，因此数据包在两个方向上都能端到端流动，而 Pod 和外部服务都不会意识到这种映射的存在。
- 出站网关：使用专门的网关来管理出站流量，提供额外的安全性和可见性。过一个或多个出站网关路由所有出站连接。网关对流量进行 SNAT（源网络地址转换），因此外部服务会看到连接来自出站网关。主要用例是提高安全性，要么通过出站网关直接执行安全角色，控制允许的流量，要么与边界防火墙（或其他外部实体）结合使用。例如，使边界防火墙看到流量来自已知的 IP 地址（出站网关），而不是来自它们不理解的动态 Pod IP 地址。

Kubernates Services：提供了一种对一组Pod的访问抽象为网络服务的方法。支持每个服务的Pod组通常使用标签选择器定义。当客户端连接到Kubernates Services服务时流量会被负载均衡到支持该服务的其中一个Pod上，Kubernates Services主要有3种类型：
- ClusterIP：这是从集群内部访问服务的常用方式。默认的服务类型是 ClusterIP。这种类型允许服务在集群内部通过一个称为服务 Cluster IP 的虚拟 IP 地址进行访问。服务的 Cluster IP 可以通过 Kubernetes DNS 发现。例如，my-svc.my-namespace.svc.cluster-domain.example。在典型的 Kubernetes 部署中，每个节点上都运行着 kube-proxy，它负责拦截到 Cluster IP 地址的流量，并在支持每个服务的 Pod 组之间进行负载均衡。在此过程中，使用 DNAT（目标网络地址转换）将目标 IP 地址从 Cluster IP 映射到选定的Pod。响应数据包在返回到发起流量的 Pod 时，会进行反向 NAT。重要的是，网络策略是基于 Pod 而不是Kubernates Services的 Cluster IP 来执行。（即，出站网络策略在 DNAT 将连接的目标 IP 更改为选定的服务 Pod 之后，对客户端 Pod 执行。由于流量的目标 IP 是唯一更改的部分，入站网络策略对 Pod 来说，原始客户端 Pod 是流量的源。）
- Node Port：这是从集群外部访问服务的常用方式。访问集群外部服务的最基本方式是使用 NodePort 类型的服务。NodePort 是在集群中每个节点上预留的一个端口，通过该端口可以访问服务。在典型的 Kubernetes 部署中，kube-proxy 负责拦截到 NodePort 的连接，并在支持每个服务的 Pod 之间进行负载均衡。访问集群外部服务的最基本方式是使用 NodePort 类型的服务。NodePort 是在集群中每个节点上预留的一个端口，通过该端口可以访问服务。在典型的 Kubernetes 部署中，kube-proxy 负责拦截到 NodePort 的流量，并在支持每个服务的 Pod 之间进行负载均衡。在此过程中，使用 NAT（网络地址转换）将目标 IP 地址和端口从节点 IP 和 NodePort 映射到选定的 Pod 和服务端口。此外，源 IP 地址从客户端 IP 映射到节点 IP，以便连接上的响应数据包通过原始节点返回，在那里可以反向 NAT。（执行 NAT 的节点拥有反向 NAT 所需的连接跟踪状态。）请注意，由于连接的源 IP 地址被 SNAT（源网络地址转换）为节点 IP 地址，服务支持 Pod 的入站网络策略无法看到原始客户端 IP 地址。这通常意味着任何此类策略仅限于限制目标协议和端口，而无法基于客户端/源 IP 进行限制。在某些场景中，可以通过使用 externalTrafficPolicy来规避这一限制，从而保留源 IP 地址。
- LoadBalancer：这是使用外部负载均衡器，作为一种更复杂的从集群外部访问服务方式。LoadBalancer 类型的服务通过外部网络负载均衡器（NLB）暴露服务。具体的网络负载均衡器类型取决于你使用的公有云提供商，或者在本地部署时，取决于与你的集群集成的特定硬件负载均衡器。服务可以通过网络负载均衡器上的特定 IP 地址从集群外部访问，默认情况下，负载均衡器会通过服务的 NodePort 在节点之间进行负载均衡。请注意，由于连接的源 IP 地址被 SNAT（源网络地址转换）为节点 IP 地址，服务支持 Pod 的入站网络策略无法看到原始客户端 IP 地址。这通常意味着任何此类策略仅限于限制目标协议和端口，而无法基于客户端/源 IP 进行限制。在某些场景中，可以通过使用 externalTrafficPolicy来规避这一限制，从而保留源 IP 地址。

广播服务IP(Advertising service IPs)：使用节点端口或网络负载均衡器的替代方法是通过 BGP 广播服务 IP 地址。这需要集群运行在支持 BGP 的底层网络上，通常意味着使用标准的顶部机架（Top of Rack）路由器进行本地部署。通过这种方式，可以在支持 BGP 的网络环境中灵活地广播服务 IP，从而实现更高效的流量管理和负载均衡。

externalTrafficPolicy: local：默认情况下，无论使用 NodePort 类型、LoadBalancer 类型的服务，还是通过 BGP 广播服务 IP 地址，从集群外部访问服务时，流量会在所有支持该服务的 Pod 之间均匀地进行负载均衡，而不考虑这些 Pod 所在的节点。可以通过将服务配置为 externalTrafficPolicy: local 来改变这种行为，这指定了流量应仅在本地节点上支持该服务的 Pod 之间进行负载均衡。当与 LoadBalancer 类型的服务或 Calico 服务 IP 地址广播结合使用时，流量仅会被引导到至少托管一个支持该服务的 Pod 的节点。这减少了节点之间的潜在额外网络跳转，更重要的是，可以保留源 IP 地址一直到 Pod，以便网络策略可以限制特定的外部客户端（如果需要）。需要注意的是，对于 LoadBalancer 类型的服务，并非所有负载均衡器都支持这种模式。在服务 IP 广播的情况下，负载均衡的均匀性取决于拓扑结构。在这种情况下，可以使用 Pod 反亲和性规则来确保支持 Pod 在拓扑结构中的均匀分布，但这会增加部署服务的复杂性。通过这种方式，externalTrafficPolicy: local 提供了一种更高效和灵活的方式来管理和控制外部流量的负载均衡，同时保留源 IP 地址以便更精细的网络策略控制。

##### eBPF

eBPF是Linux的一项功能，允许快速且安全的将程序加载到内核中，以自定义其操作。eBPF是嵌入在Linux内核中的一种虚拟机。它允许将程序加载到内核当中，并附加到钩子(hook)上，这些钩子在某些事件发生时被触发，这使得可以自定义内核的行为，虽然每种类型的钩子都使用相同的eBPF虚拟机，但钩子的功能却差异很大。将程序加载到内核当中很可能是危险的，内核会通过一个非常严格的内核验证器运行加载的所有程序，验证器会对程序进行沙盒化。确保只能访问允许访问的内存部分，并且确保它快速终止。eBPF 代表“**扩展的伯克利包过滤器**”（extended Berkeley Packet Filter），伯克利包过滤器是一种早期的虚拟机，专门用于过滤数据包。例如，tcpdump 等工具使用这种“经典”的 BPF 虚拟机来选择应发送到用户空间进行分析的数据包，eBPF 是 BPF 的大幅扩展版本，适用于内核中的通用用途。虽然名称保留了下来，但eBPF可以用于很多用途。

eBPF的程序类型，eBPF 程序的功能在很大程度上取决于它所附加的钩子：
- 跟踪程序：可以附加到内核中的大量函数上，跟踪程序对于收集统计数据和深入调试内核非常有用。大多数跟踪钩子只允许对函数处理的数据进行只读访问，但有些允许修改数据。
- 流量控制(tc)程序：可以附加到给定网络设备的入口和出口，内核为每个数据包执行一次程序。由于这些钩子用于数据包处理，内核允许程序修改或扩展数据包、丢弃数据包、标记数据包以进行排队，或将数据包重定向到另一个接口。
- XDP(eXpress Data Path)：它是是 eBPF 钩子的名称，每个网络设备都有一个XDP入口钩子，它在内核为数据包分配套接字缓冲区之前，为每个传入的数据包触发一次。XDP可以为DoS保护和入口负载均衡等场景提供卓越的性能。XDP的缺点是需要网络驱动程序的支持才能获得良好的性能。XDP本身不足以实现Kubernates Pod网络所需的所有逻辑，但XDP和流量控制钩子的组合效果很好。
- 套接字程序：允许eBPF程序更改新创建套接字的目标IP或强制套接字绑定到正确的源IP地址。
- 安全相关钩子：允许以各种方式对程序行为进行策略控制。例如，seccomp 钩子允许以细粒度的方式对系统调用进行策略控制。

内核通过“辅助函数”暴露每个钩子的功能。例如，tc 钩子有一个辅助函数来调整数据包大小，但该辅助函数在跟踪钩子中是不可用的。使用 eBPF 的一个挑战是不同的内核版本支持不同的辅助函数，缺乏某个辅助函数可能使得无法实现特定功能。附加到 eBPF 钩子的程序能够访问 BPF “maps”。BPF maps 有两个主要用途：允许 BPF 程序存储和检索长期存在的数据。允许 BPF 程序与用户态程序之间进行通信。BPF 程序可以读取用户态写入的数据，反之亦然。

##### Kubernetes Resources

为命名空间设置默认内存请求和限制：如果你的命名空间设置了内存”资源配额“（资源配额提供了限制每个命名空间的资源消耗总和的约束），那么为内存限制设置一个默认值会很有帮助。以下是内存资源配额对命名空间施加的三条限制：
- 命名空间中运行的每个Pod中的容器都必须有内存限制（如果Pod中的每个容器声明了内存限制，Kubernates可以通过将其容器的内存限制相加推断出Pod级别的内存限制）。
- 内存限制用于在Pod被调度到的节点上执行资源预留。预留给命名空间中所有Pod使用的内存总量不能超过规定的限制。
- 命名空间中所有Pod实际使用的内存总量也不能超过规定的限制。

如果该命名空间中的任何Pod的容器未指定内存限制，控制面将默认内存限制应用于该容器，这样Pod可以在受到内存资源配额(ResourceQuota)限制的命名空间中运行。

为命名空间设置默认的CPU请求和限制：一个Kubernates集群可以划分为多个命名空间。如果你在具有默认CPU限制的命名空间内创建一个Pod，并且这个Pod中所有容器都没有声明自己的CPU限制，那么控制面会为容器设定默认的CPU限制。如果命名空间设置了资源配额，为CPU限制设置一个默认值会很有帮助。以下是CPU资源配额对命名空间施加的两条限制：
- 命名空间中运行的每个Pod中的容器都必须有CPU限制。
- CPU限制用于在Pod被调度到的节点上执行资源预留。

预留给命名空间中所有的Pod使用的CPU总量不能超过规定的限制。如果该命名空间中的任何 Pod 的容器未指定 CPU 限制， 控制面将默认 CPU 限制应用于该容器， 这样 Pod 可以在受到 CPU 资源配额限制的命名空间中运行。

配置命名空间的最小和最大内存约束：为命名空间定义一个有效的内存资源限制范围，在该命名空间中每个新创建 Pod 的内存资源是在设置的范围内。

为命名空间配置内存和 CPU 配额：
- 在该命名空间中的每个 Pod 的所有容器都必须要有内存请求和限制，以及 CPU 请求和限制。
- 在该命名空间中所有 Pod 的内存请求总和不能超过命名空间的内存请求。
- 在该命名空间中所有 Pod 的内存限制总和不能超过命名空间的内存限制。
- 在该命名空间中所有 Pod 的 CPU 请求总和不能超过命名空间的CPU请求。
- 在该命名空间中所有 Pod 的 CPU 限制总和不能超过该命名空间的CPU限制。

k8s 更改 CPU 管理器策略：在Kubernates中，更改CPU管理策略主要通过修改kubelet的配置参数 --cpu-manager-policy 或Kubelet或KubeletConfiguration的cpuManagerPolicy字段来实现。CPU管理支持两种策略：
- none：默认策略，不启用特殊的CPU亲和性管理，使用CFS配额限制CPU。
- static：为节点上符合条件的Guaranteed Pod提供CPU独占和亲和性，减少CPU迁移和上下文切换，提高性能。

更改 CPU 管理器策略的步骤：由于CPU管理策略只能在kubelet生成新的Pod时生效，简单的更改配置不会影响现有Pod：
- 腾空节点：将节点上所有Pod停止，确保节点空闲。
- 停止kubelet服务：在节点上停止kubelet避免配置冲突。
- 删除旧的CPU管理器状态文件：默认路径为/var/lib/kubelet/cpu_manager_state。删除该文件可以清除旧策略的状态，避免与新策略冲突。
- 修改kubelet配置：通过编辑kubelet文件（如 ConfigMap 或 kubelet 启动参数）将cpuManagerPolicy设置为所需策略（如static）。
- 重启kubelet服务：启动kubelet，使新配置生效。重复以上步骤对每个节点进行操作。

参数说明：--cpu-manager-policy：指定 CPU 管理策略，支持 none 和 static。--cpu-manager-reconcile-period：CPU 管理器状态同步频率，默认与节点状态更新频率相同。--cpu-manager-policy-options：用于微调 static 策略的行为，需开启相关特性门控。通过以上步骤，可以安全地将 CPU 管理器策略从 none 切换到 static，实现更精细的 CPU 绑定和管理，从而提升对 CPU 亲和性敏感工作负载的性能表现。

拓扑管理器(Topology Manager)：是kubelet的一个组件，旨在协调CPU、内存和设备管理等多个资源管理组件，使资源分配能够考虑节点的硬件拓扑结构，以优化性能并减少延迟。在启用拓扑管理器(Topology Manager)之前，Kubernates中的CPU管理器和设备管理器等组件独立做资源分配，，可能导致CPU和设备分配在不同的NUMA节点，增加跨节点访问的延迟，影响性能。拓扑管理器(Topology Manager)作为”事实来源“，收集各Hint Provider（提示提供者）发来的拓扑信息，统一协调资源分配，使Pod的CPU、内存和设备资源尽可能在同一或相邻的NUMA节点上分配，从而提升性能。工作原理：
- 拓扑管理器(Topology Manager)接收Hint Providers发送的NUMA节点位掩码和首选分配信息。
- 根据配置的策略，对提示进行处理，选出最优的拓扑分配方案。
- 根据选定的提示，决定是否接收或拒绝Pod在该节点上运行。
- 选定的提示会存储起来，供 Hint Providers在后续资源分配时参考。

支持的拓扑管理策略(Topology Manager Policies)，拓扑管理器(Topology Manager)提供了四种策略，通过kubelet参数--topology-manager-policy配置：
- none（默认）：不进行任何拓扑对齐，资源分配不考虑NUMA拓扑，Pod不拒绝。
- best-effort：收集Hint Providers的拓扑信息，尝试对齐资源，但如果无法满足首选拓扑，仍然允许Pod调度，Pod不拒绝。
- restricted：只有当所有资源都满足首选拓扑时才允许Pod调度，否则拒绝Pod。Pod 进入 Terminated 状态，调度失败。
- single-numa-node：资源必须全部分配在同一个NUMA节点上，否则拒绝Pod。Pod 进入 Terminated 状态，调度失败。
- prefer-closest-numa-nodes（额外需开启 TopologyManagerPolicyOptions 功能门控）：优先选择距离更近的 NUMA 节点，减少跨节点访问延迟（Kubernetes 1.32 及以后默认可用）。
- max-allowable-numa-nodes（额外需开启 TopologyManagerPolicyOptions 功能门控）：限制允许的最大 NUMA 节点数，避免过多 NUMA 跨越带来的性能损失。

其中，restricted 和 single-numa-node 策略对性能要求较高的应用更友好，因为它们严格保证了资源的 NUMA 本地性。启用条件：节点必须启用CPU管理器的static策略。Pod需要是Guaranteed QoS类别，确保资源请求严格匹配。kubelet需要开启拓扑管理器(Topology Manager)功能（Kubernetes 1.18 及以后默认开启）。综上，拓扑管理器(Topology Manager)通过协调CPU、设备和内存管理器的拓扑提示，给予配置的策略，在节点层面实现资源的NUMA本地性对齐，提升性能和降低延迟，尤其适合对延迟和吞吐有严格要求的高性能计算、机器学习、金融服务等场景。

NUMA（Non-Uniform Memory Access，非一致内存访问）是一种用于多处理器系统的内存架构，旨在解决传统 SMP（对称多处理器）架构中，因所有处理器共享同一内存总线而导致的性能瓶颈问题。NUMA 架构的关键概念：
- 节点(Node)：NUMA架构将系统划分为多个节点，每个节点包含一个或多个处理器，本地内存和I/O设备，节点在物理上彼此独立，并通过高速互联网络连接在一起，形成一个整体系统。
- 本地内存和远程内存：处理器可以快速访问其所在节点的本地内存，而访问其他节点的远程内存则会有延迟，访问本地节点的速度最快，访问远端节点的速度最慢，访问速度与节点距离有关。
- 节点距离(Node Distance)：用于衡量CPU访问不同节点内存的速度差异，距离越远访问速度越慢。

NUMA优势：解决内存访问瓶颈，通过将内存分配到各个节点，使处理器优先访问本地内存，降低了内存访问延迟，提高了多处理器系统的性能。提高系统的扩展性，NUMA架构简化了总线的设计，可以支持更多的处理器，提高系统的扩展性。NUMA 架构通过将 CPU 分组到不同的节点，并为每个节点配置本地内存，使得 CPU 优先访问本地内存，从而降低内存访问延迟，提高系统整体性能. 在 Kubernetes 等云原生环境中，NUMA 亲和性调度等技术可以被用来优化任务调度和内存分配策略，进一步提高内存访问效率和整体性能。

Kubernates自定义 DNS 服务：主要是指在集群内部配置和管理DNS解析服务，以满足集群内Pod和Service的名称解析需求，同时支持对特定域名使用自定义的DNS服务器进行解析。Kubernetes 集群内置 DNS 服务用于实现服务发现，允许 Pod 通过服务名访问其他服务，而不必使用 IP 地址。CoreDNS以Deployment形式运行，通常暴漏为Kubernates服务（服务名为 kube-dns），Pod通过kubelet的--cluster-dns参数配置使用该DNS服务。DNS 服务支持正向查找（A、AAAA 记录）、SRV 记录、PTR 记录等多种查询类型。自定义DNS服务满足集群内部对特定域名的解析需求，例如某些域名请求转发到自建的DNS服务器。支持集群外部域名解析的定制，比如使用特定的上游DNS服务器（如 Google DNS）。允许Pod使用自定义的DNS配置，覆盖默认的集群DNS设置。自定义DNS配置方式：CoreDNS 的配置通过 ConfigMap 管理，管理员可以在 ConfigMap 中添加自定义的 DNS 解析规则。例如，可以为特定域名（如 a.test.com）配置专门的 DNS 服务器：
```json
a.test.com:53 {
  errors
  cache 30
  forward . 10.10.10.1
}
```
Pod 级别的自定义 DNS 配置：Kubernetes 允许在 Pod 规范中通过 dnsPolicy 和 dnsConfig 字段自定义 DNS 行为。常用的 dnsPolicy 有：ClusterFirst（默认）-优先使用集群 DNS；Default -继承宿主机的 DNS 配置；None -忽略默认 DNS，完全使用自定义的 dnsConfig 配置。通过 dnsConfig 可以指定：nameservers -自定义 DNS 服务器 IP 列表（最多3个）；searches -DNS 搜索域列表。options -DNS 选项，如 ndots 等。
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: dns-example
spec:
  containers:
  - name: test
    image: nginx
  dnsPolicy: "None"
  dnsConfig:
    nameservers:
    - 192.0.2.1
    searches:
    - ns1.svc.cluster-domain.example
    - my.dns.search.suffix
    options:
    - name: ndots
      value: "2"
    - name: edns0
```
该 Pod 内部的 /etc/resolv.conf 会包含上述自定义的 DNS 配置。DNS查询流程：
- Pod 发起 DNS 查询后，默认会先查询 CoreDNS缓存。
- 如果查询的域名匹配集群域名后缀（如 .cluster.local），由 CoreDNS 直接解析。
- 如果查询匹配自定义存根域，则转发到对应的自定义 DNS 服务器。
- 其他查询则转发到上游 DNS 服务器（如 Google DNS）。

k8s 的自定义 DNS 服务主要通过配置CoreDNS的ConfigMap来实现对特定域名的定制解析，同时支持 Pod 级别的 DNS 策略和配置，满足灵活多样的 DNS 解析需求，保障集群内外服务的顺畅访问和集成。

Kubernetes NetworkPolicy是一种用于控制Pod与其它网络实体之间通信的资源对象，主要目的是实现基于标签选择器的网络访问控制，增强集群的安全性和隔离性。NetworkPolicy以Pod为中心，通过标签选择器（PodSelector）选择一组Pod定义允许哪些流量可以进入(Ingress)或离开(Egress)这些Pod。工作在网络层(L3/L4)，控制IP地址和端口的访问权限，不涉及应用层协议。NetworkPolicy需要底层网络插件支持，如Flannel、Calico、Cilium、Weave Net等，否则策略不会生效。如果没有任何NetworkPolicy，Pod之间的网络通信默认是允许的，一旦在命名空间中创建了NetworkPolicy，只有明确允许的流量才会被放行。PodSelector(选择受策略约束的Pod)、PolicyType（定义策略类型，常见有Ingress -入站流量控制、Egress -出站流量控制，可以单独或同时使用），定义允许的来源（Ingress）和目的地（Egress），可以通过以下三种方式指定：podSelector -选择特定标签的 Pod；namespaceSelector -选择特定标签的命名空间内的Pod；ipBlock -指定允许或拒绝的IP网段（CIDR），端口和协议：可以指定允许访问的端口号和协议（TCP、UDP、SCTP）。NetworkPolicy声明示例：
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: example-network-policy
  namespace: default
spec:
  podSelector:
    matchLabels:
      role: db
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - ipBlock:
        cidr: 172.17.0.0/16
        except:
        - 172.17.1.0/24
    - namespaceSelector:
        matchLabels:
          project: myproject
    - podSelector:
        matchLabels:
          role: frontend
    ports:
    - protocol: TCP
      port: 6379
  egress:
  - to:
    - ipBlock:
        cidr: 10.0.0.0/24
    ports:
    - protocol: TCP
      port: 5978
```
该策略表示：作用于 default 命名空间中带有role=db标签的Pod。允许来自指定 IP 段（排除部分网段）、特定命名空间和特定标签Pod的入站访问，且只允许 TCP 6379 端口。允许这些 Pod 访问指定 IP 段的 TCP 5978 端口。应用场景：限制服务A的Pod不能访问服务B的Pod。不同租户命名空间之间实现网络隔离。对暴露到外部的Pod实现白名单访问。限制Pod只能访问指定端口或指定来源。NetworkPolicy只影响被选中Pod的网络流量。Pod 默认可以访问节点和集群 DNS 等基础设施，除非特别限制。多个 NetworkPolicy 规则是“或”关系，满足任一规则即可放行流量。需要确保所用的网络插件支持 NetworkPolicy 功能。Kubernetes 的 NetworkPolicy 是一种强大的网络访问控制机制，通过声明式的 YAML 配置，结合标签选择器和 IP 范围，实现对 Pod 网络流量的细粒度管理，提升集群安全性和网络隔离能力。

Kubernates云控制器管理器(Cloud Controller Manager，简称CCM)旨在将集群连接到云供应商的应用程序编程接口(API)，并将与云平台交互的组件分离开来。由于云驱动的开发和发布与Kubernates项目本身步调不同，将特定于云环境的代码抽象到云控制管理器，当服务类型设定为Type=LoadBalancer时，云控制器管理器(CCM)会为该服务创建配置传统型负载均衡(CLB)或网络型负载均衡(NLB)，包括CLB、NLB、监听、后端服务器组等资源。当服务对应的后端Endpoint或者集群节点发生变化时，云控制器管理器(CCM)会自动更新CLB或NLB的后端虚拟服务组。当集群网络组件为Flannel时，云控制器管理器(CCM)组件负责打通容器与节点的网络，实现容器跨节点通信。云控制器管理器(CCM)会将节点的Pod网段信息写入到VPC的路由表中，从而实现跨节点的网络通信。该功能无需配置，安装即可使用。开发云控制器管理器(CCM)有两种方式：
- 树外(Out of Tree)：使用满足cloudprovider.Interface接口的实现来创建一个GO语言包，使用来自Kubernates核心代码库cloud-controller-manager中的main.go作为模版，在main.go中导入云包，确保包有一个init块来运行cloudprovider.RegisterCloudProvider。
- 树内(In Tree)：对于树内驱动，可以将树内云控制器管理器(CCM)作为集群中的DaemonSet来运行。

静态数据加密：是指数据写入存储介质(ETCD)时进行加密，确保数据在磁盘上是加密状态，即使存储介质被非法访问，数据也无法被直接读取。Kubernates支持对API资源数据进行加密，最常见的是对Secret资源进行加密，除了Secret，也可以配置其他资源类型（如自定义资源），但需要集群版本支持。Kubernates API Server通过启动参数 --encryption-provider-config指定一个加密配置文件，改配置文件定义了加密策略和秘钥，常用的加密方式包括：AES-CBC、AES-GCM、KMS v1、KMS v2等。当有数据写入ETCD时，API Server根据配置对数据进行加密，读取时自动解密，保证数据在ETCD中是加密存储的。配置步骤：
- 生成加密密钥：通常生成一个32字节的随机密钥，并进行base64编码。
- 编写加密配置文件：YAML格式，指定资源类型（如secrets）和加密提供者（如aescbc）及密钥。
- 修改API Server启动参数：添加--encryption-provider-config指向配置文件路径。
- 重启API Server：使配置生效。
- 验证加密效果：通过etcdctl命令查看存储的Secret是否为加密格式，且通过kubectl命令能正常读取解密后的数据。

Kubernetes静态数据加密通过配置API Server加密提供者，实现对Secret等敏感资源在etcd中的加密存储，防止数据泄露，增强集群安全性。配置过程包括生成密钥、编写加密配置、修改API Server参数及重启等步骤，且需要对已有数据进行迁移加密。此功能是保护Kubernetes集群中敏感信息的重要安全措施。

Kubernetes中的“关键附加组件Pod的保证调度”机制主要目的是确保那些对集群正常运行至关重要的附加组件Pod能够被优先调度和持续运行，避免因资源紧张或节点变动导致它们被驱逐后无法及时恢复，从而保证集群的稳定性和功能完整性。Kubernetes核心组件（如API服务器、调度器、控制器管理器）通常运行在控制平面节点上。但某些关键附加组件（如metrics-server、DNS、UI等）必须运行在普通的集群节点上。如果这些关键附加组件Pod被驱逐（无论是手动驱逐还是升级等操作的副作用），且无法及时重新调度，集群功能可能会受到严重影响甚至停止工作。如何标记Pod为关键（Critical）：
- 关键Pod必须运行在kube-system命名空间（该命名空间可通过参数配置）。
- 需要为Pod设置priorityClassName，值为system-cluster-critical或system-node-critical，其中system-node-critical优先级最高，甚至高于system-cluster-critical。
- 关键Pod还需带有CriticalAddonsOnly的toleration，以配合调度器的保证调度机制。

保证调度的实现机制：当集群资源紧张，调度器发现没有节点有足够资源调度关键Pod时，Rescheduler组件会介入。Rescheduler尝试通过驱逐一些非关键的Pod来释放资源，为关键Pod腾出空间。在驱逐过程中，为避免其他Pod抢占腾出的资源，目标节点会临时添加一个名为CriticalAddonsOnly的污点（taint），只有带有对应容忍（toleration）的关键Pod才能被调度到该节点。一旦关键Pod成功调度，该污点会被移除。标记为关键的Pod并非绝对不可被驱逐，静态Pod标记为关键时无法被驱逐，但非静态Pod即使被驱逐也会被重新调度，保证不会永久不可用。当前Rescheduler没有保证选择哪个节点以及驱逐哪些Pod，因此启用该机制时，普通Pod可能会被偶尔驱逐以保证关键Pod调度。Rescheduler默认作为静态Pod启用，可以通过修改启动参数或删除其manifest文件来禁用。

Kubernetes的关键附加组件Pod保证调度机制通过优先级（priorityClassName）和调度器的资源调度策略，结合Rescheduler的驱逐释放资源策略，确保关键附加组件Pod在资源紧张时依然能被优先调度和保持运行，保障集群的核心功能稳定性和可用性。将关键Pod标记为system-cluster-critical或system-node-critical优先级。关键Pod运行在kube-system命名空间。使用Rescheduler驱逐非关键Pod释放资源。通过CriticalAddonsOnly污点和容忍机制保护关键Pod调度空间。这样设计有效避免了关键附加组件因资源竞争而长时间不可用的风险，是Kubernetes集群稳定运行的重要保障机制。

Kubernates中的IP伪装代理(ip-masq-agent)是一个DaemonSet形式部署在每个节点上的代理程序，主要用于管理节点上的IP伪装规则，确保容器流量在发送到集群外部地址时使用节点的IP地址作为源地址，而非Pod的IP地址，从而满足某些网络环境对流量源地址的要求。IP伪装(Masquerade)：通过配置iptables规则，将Pod发往集群节点之外的流量源IP地址伪装成节点IP隐藏Pod的真实IP，保证流量能被外部网络接收。非伪装CIDR范围：默认情况下，ip-masq-agent将RFC 1918定义的三大私有IP段（10.0.0.0/8、172.16.0.0/12、192.168.0.0/16）和链路本地地址段（169.254.0.0/16）视为非伪装范围，即这部分流量不会被伪装。iptables链：代理创建一个名为IP-MASQ-AGENT的iptables链，并在POSTROUTING链中跳转到该链，判断流量是否需要伪装。配置自动重载：ip-masq-agent会每隔默认60秒（可配置）从/etc/config/ip-masq-agent配置文件重新加载规则，实现动态更新。部署与配置：
- 部署方式：通过kubectl应用官方提供的DaemonSet YAML文件部署ip-masq-agent。
- 节点标签：需要给希望运行ip-masq-agent的节点打标签（如node.kubernetes.io/masq-agent-ds-ready=true），使DaemonSet只在指定节点运行。
- 配置文件：配置文件支持YAML或JSON格式，主要包含以下可选字段：nonMasqueradeCIDRs：指定不需要伪装的IP地址范围列表。masqLinkLocal：是否对链路本地地址段进行伪装，默认开启（true）。resyncInterval：配置文件重新加载间隔时间，支持秒（s）或毫秒（ms）。
- 自定义配置：通过创建ConfigMap并挂载到ip-masq-agent容器，实现对默认伪装规则的定制，比如只对特定IP段进行伪装。

使用场景：在某些云环境中，外部流量必须来自于节点的IP地址，Pod的地址直接访问会被拒绝，此时需要ip-masq-agent进行伪装。集群内Pod访问集群外服务时，确保流量原地址为节点IP，满足安全和网络策略的要求。控制哪些IP范围流量需要伪装，避免不必要的SNAT，提升网络性能和可控性。

Kubernates中闲置存储资源消耗(Limit Storage Consumption)主要有两种机制来实现：LimitRange、ResourceQuota，它们可以帮助集群管理员来控制单个Pod或命名空间的存储请求和实际使用量，防止存储资源被过度占用，保障集群稳定运行。
- LimitRange（限制范围）：在命名空间级别定义Pod或容器的资源请求和限制的默认值和最大值，包括存储资源（如临时存储ephemeral-storage）。限制单个容器或Pod请求的存储资源大小，防止用户请求资源过大导致浪费。可以设置默认请求和限制，避免用户忘记设置资源限制。
```yaml
apiVersion: v1
kind: LimitRange
metadata:
  name: storage-limit-range
  namespace: example-namespace
spec:
  limits:
  - type: Container
    max:
      ephemeral-storage: "2Gi"
    min:
      ephemeral-storage: "100Mi"
    default:
      ephemeral-storage: "500Mi"
    defaultRequest:
      ephemeral-storage: "200Mi"
```
以上配置限制单个容器的临时存储最大为2Gi，最小为100Mi，默认请求为200Mi，默认限制为500Mi。
- ResourceQuota（资源配额）：对命名空间中的所有资源的总消耗设置上限，包括存储资源的从请求量和使用量。限制整个命名空间中持久卷声明(PVC)请求的存储总量。限制临时存储的总请求量。防止整个名空间无限制的创建大量存储资源。导致集群资源枯竭。
```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: storage-quota
  namespace: example-namespace
spec:
  hard:
    requests.storage: "10Gi"
    persistentvolumeclaims: "5"
```
以上配置限制该命名空间中所有PVC请求的存储总和不超过10Gi，PVC数量不超过5个。
- 结合使用：LimitRange控制单个Pod/容器的存储请求和限制，ResourceQuota控制命名空间整体的存储资源消耗。通过两者结合，既防止单个应用消耗过多存储，也防止整个命名空间超出资源预算。

Kubernates支持对临时存储(ephemeral-storage)和持久存储(requests.storage)分别进行限制。设置合理的存储限制有助于提升集群的资源利用率和稳定性，避免因存储资源耗尽导致Pod被驱逐或调度失败。监控存储资源使用情况，结合LimitRange和ResourceQuota调整策略，是存储资源管理的最佳实践。Kubernetes通过LimitRange和ResourceQuota机制限制存储资源的请求和消耗，帮助管理员控制单个Pod和命名空间的存储使用量，防止资源滥用，保障集群稳定和高效运行。

复制控制平面并迁移至云控制器管理器：是指将原本内嵌在kube-controller-manager中的云控制器，迁移到独立的cloud-controller-manager组件中运行的过程。Kubernates的kube-controller-manager中包含了云平台相关的控制器（如路由管理、服务管理、节点生命周期管理等），随着云提供商的代码被逐渐剥离，这些云平台相关的控制器必须迁移到独立的cloud-controller-manager中。这样做能让云提供商独立开发和更新其控制器，而不必等待Kubernetes主项目的版本发布。迁移的核心机制——Leader Migration（领导者迁移）：
- Leader Migration是一种机制，支持高可用(HA)控制平面的环境下，云相关控制器在kube-controller-manager和cloud-controller-manager之间安全迁移。通过共享的资源锁，实现控制器领导权的切换，确保同一时间只有一方管理某个云控制器，避免冲突。迁移期间，两个组件同时运行，但控制器只会在一个组件中活跃。Leader Migration需要在两个组件上分别开启--enable-leader-migration参数，并配置迁移相关的配置文件(LeaderMigrationConfiguration)指定哪些控制器迁移到那个组件。迁移步骤：
- 准备阶段：确保控制平面运行kube-controller-manager且启用领导者选举（Leader Election）。确认云提供商支持cloud-controller-manager且实现了Leader Migration。授权kube-controller-manager访问迁移租约（Lease）资源。
- 配置Leader Migration：创建LeaderMigrationConfiguration配置文件，定义当前控制器归属（如route、service、cloud-node-lifecycle仍在kube-controller-manager）。修改kube-controller-manager的启动参数，挂载配置文件并启用Leader Migration。
- 部署Cloud Controller Manager：在升级版本（N+1）中，部署cloud-controller-manager，配置相同的Leader Migration配置文件，但将云控制器的归属改为cloud-controller-manager。将kube-controller-manager的--cloud-provider参数改为external，且不再启用Leader Migration。
- 滚动升级控制平面节点：逐个替换控制平面节点，从旧版本（仅运行kube-controller-manager）切换到新版本（运行kube-controller-manager和cloud-controller-manager）。迁移租约确保控制器领导权在两个组件间安全切换，避免冲突。直到所有节点升级完成，控制器完全迁移到cloud-controller-manager。
- 回滚支持：如果需要回滚，可以将旧版本节点重新加入集群，启用Leader Migration，逐步替换新版本节点。

适用场景：需要升级Kubernetes控制平面版本，同时迁移云提供商控制器到外部Cloud Controller Manager。运行高可用控制平面且不能容忍控制器管理组件停机的生产环境。云提供商已发布支持Leader Migration的外部Cloud Controller Manager。单节点控制平面或能容忍控制器停机的场景，可以跳过Leader Migration，直接切换。迁移过程中，确保RBAC权限允许访问迁移锁Lease资源。迁移配置文件支持指定具体控制器的归属，也可使用通配符*简化配置。迁移完成后，kube-controller-manager不再运行云相关控制器，cloud-controller-manager承担全部云控制职责。Kubernates的复制控制平面并迁移至云控制器管理器是一个通过Leader Migration机制。逐步将云相关控制器从kube-controller-manager迁移到独立的cloud-controller-manager的过程。此举实现了云控制器与核心控制器的解耦，提升了云提供商的灵活性和Kubernetes架构的模块化，是现代云原生集群管理的重要升级路径。

Kubernates中的ETCD集群是Kubernates集群的核心数据存储后端，是一个一致性强、高可用的分布式键值存储系统，ETCD在Kubernates中的作用：ETCD用作Kubernates所有集群数据的后台数据库，存储集群状态、配置信息、服务返现数据等。它通过Raft一致性算法保证数据的强一致，确保集群状态的准确和可靠。Kubernates组件(kube-apiserver)通过访问ETCD来获取或更新集群状态。

ETCD集群的特性：ETCD是一个基于Leader的分布式系统，集群中有一个主节点负责协调（Master），其他节点为从节点(Follower)。集群成员数量应为奇数，以保证选举机制的正常运行和数据一致性。ETCD集群对网络和磁盘I/O非常敏感，资源不足会导致心跳超时，进而导致集群不稳定，影响Kubernates调度和运行。生产环境中应保证 ETCD运行在资源充足的专用机器或隔离环境，避免资源竞争导致集群不稳定。定期备份 ETCD 数据，确保在故障时可以恢复集群状态。 ETCD 集群成员的动态添加和移除需要谨慎操作，避免影响集群稳定。对于大型 Kubernetes 集群，可以考虑对 ETCD 数据进行水平拆分，例如将 Pod 资源数据单独存储在一个 ETCD 集群中，以降低单个  ETCD 集群的负载和请求延迟。这种拆分需要谨慎规划和实施，避免影响数据一致性和集群稳定性。

Kubernetes 中操作 ETCD 集群涉及集群的部署（单节点和多节点）、安全访问配置、性能资源保障以及定期备份恢复等方面。ETCD 是 Kubernetes 集群稳定和高可用的关键组件，合理配置和维护 ETCD 集群对整个 Kubernetes 集群的健康至关重要。

Kubernates访问集群主要是通过Kubernates API实现的，API是Kubernates的核心接口，允许用户查询、修改集群状态。访问Kubernates集群的前提：你需要有一个运行中的Kubernates集群，需要有访问该集群的凭据（如 kubeconfig 文件），通常有集群管理员提供。本地安装并配置好kubectl命令行工具，它是访问Kubernates API最常用的客户端工具。可以通过命令kubectl config view查看当前配置的集群地址和凭据。访问Kubernates API的方式：
- 使用 kubectl 命令行工具（推荐）。
- 直接访问 REST API。
- 使用客户端库。

访问Kubernates API需要认证授权：常用的认证方式包括客户端证书、Bearer Token、认证代理等。认证成功后，API服务器会根据授权策略（如RBAC）决定是否允许访问特定资源和操作。常见的授权模式有RBAC、ABAC、Webhook授权等。访问 Kubernetes 集群主要通过 Kubernetes API，最常用和推荐的方式是使用 kubectl，它自动处理连接和认证。也可以直接调用 REST API，但需要手动管理认证信息。访问时必须通过认证和授权，确保集群安全。远程访问 Kubernetes API 需要额外配置安全机制和网络访问策略。

Kubernates中为系统守护进程预留计算资源是指在节点资源管理中，专门为操作系统何Kubernates系统组件（守护进程）预留一定的CPU、内存和存储资源。避免这些系统进程与运行的Pod争夺资源，导致节点资源不足甚至系统不稳定的问题。Kubernetes 节点的所有资源（Capacity）都可以被 Pod 使用，这会带来风险：节点上运行的操作系统守护进程（如 sshd、udev）和 Kubernetes 系统组件（如 kubelet、container runtime）也需要资源，如果没有预留，Pod 可能会抢占这些资源，导致系统进程被杀死或节点状态异常，进而影响整个集群的稳定性。为了解决上述问题，Kubernates的kubelet提供了一个叫做Node Alloctable的特性，用于为系统守护进程和Kubernates组件预留资源，这样，即使节点负载很高，Pod也不会占用预留给系统进程的资源，避免资源争夺导致的节点不稳定。资源预留分类：
- kube-reserved：为Kubernates系统守护进程预留资源，包含kubelet、container runtime、node problem detector等，但不包含以Pod形式运行的系统组件。
- system-reserved：为操作系统的守护进程预留资源，如sshd、udev等，同时建议预留内核内存和用户登录会话资源。
- eviction-threshold：为节点驱逐机制预留的资源阈值，防止节点过载时触发OOM杀死关键进程。

支持预留的资源类型包括CPU、内存、临时存储和进程ID数量。节点资源分配可以理解为：
```bash
Node Capacity（节点总资源）
  ├─ kube-reserved（Kubernetes 系统进程预留）
  ├─ system-reserved（操作系统守护进程预留）
  ├─ eviction-threshold（驱逐阈值预留）
  └─ allocatable（可供Pod使用的资源）
```
计算公式为：allocatable = NodeCapacity − kube−reserved − system−reserved − eviction−threshold，Pod 调度时参考的是 allocatable 资源，而非节点总资源。在 kubelet 启动参数或配置文件中设置：--kube-reserved=cpu=100m,memory=100Mi,ephemeral-storage=1Gi,pid=1000、--system-reserved=cpu=100m,memory=100Mi,ephemeral-storage=1Gi,pid=1000，还可以指定对应的 cgroup：--kube-reserved-cgroup=/kubelet.service、--system-reserved-cgroup=/system.slice。这样 kubelet 会对这些守护进程的资源使用做硬限制，确保预留资源不被 Pod 占用。

最佳实践：
- 根据节点上实际运行的系统守护进程和 Kubernetes 组件的资源需求，合理配置 kube-reserved 和 system-reserved。
- 确保对应的 cgroup 已经创建，避免 kubelet 启动失败。
- 结合 eviction-threshold 设置，防止节点资源耗尽导致系统进程被杀。
- 对于需要高性能或特殊场景（如电信/NFV），可以通过指定 CPU 集合（cpuset）和中断绑定，进一步隔离系统守护进程和工作负载。

Kubernetes 的系统守护进程资源预留功能通过 Node Allocatable 特性，确保系统和 Kubernetes 组件有足够资源稳定运行，避免 Pod 资源抢占导致节点不稳定，是生产环境集群稳定性的重要保障。

在Kubernates中，安全排空(Drain)节点是指对节点进行维护（如升级内核、硬件维护、下线节点等）之前，优雅地将节点上的所有Pod驱逐(evict)出去，确保应用不中断或尽量减少中断，从而保证集群的稳定性和服务的连续性。安全排空节点(Safely Drain a Node)：使用kubectl drain命令可以将指定节点标记为不可调度(unschedulable)组织新Pod调度到该节点，同事驱逐该节点上所有可驱逐的Pod（即非 DaemonSet 管理的 Pod 和非镜像 Pod），为节点维护腾出资源。如果直接关闭节点而不先排空，节点上的Pod会突然终止，导致服务中断，Kubernates会尝试在其他节点重新调度这些Pod，但这会带来延迟或潜在的服务不可用。通过kubectl drain，可以优雅的通知Kubernates迁移Pod，减少影响。

安全排空节点的步骤：
- 标记节点为不可调度（Cordon）：使用命令，kubectl cordon <node-name>。该操作阻止新的 Pod 调度到该节点，但不会影响已经运行的 Pod。
- 驱逐节点上的 Pod（Drain）：使用命令，kubectl drain <node-name> --ignore-daemonsets --delete-emptydir-data。--ignore-daemonsets：忽略由 DaemonSet 管理的 Pod，因为它们会自动重新创建，无法被驱逐。--delete-emptydir-data：允许删除使用了 emptyDir 临时存储的 Pod，提醒数据会丢失。该命令会逐个驱逐节点上的 Pod，等待它们优雅终止，完成后节点即空闲。
- 确认驱逐完成：使用命令，kubectl get pods --all-namespaces -o wide，确认该节点上没有非 DaemonSet 的 Pod 运行。
- 维护完成后恢复节点调度：使用命令，kubectl uncordon <node-name>，使节点重新可调度，恢复正常使用。

- 基于Kubernetes的机器学习工作负载：Kubeflow
- 基于Kubernetes的无服务器环境：Apache OpenWhisk、Fission、Kubeless、nuclio和OpenFaaS
- 基于Kubernetes构建的数据平台：Iguazio（数据流式分析）
- 基于Kubernates的网络插件：Cilium、Flannel、Calico