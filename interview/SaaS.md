
#### 美团餐饮SaaS平台

美团餐饮SaaS平台为餐饮企业提供餐厅数字化解决方案（一站式IT解决方案），帮餐厅实现从前厅管理、后厨生产管理、会员管理、线上运营管理、供应链管理到外卖的数字化经营。美团餐饮系统实现了餐厅和平台的打通，帮助餐厅连接顾客，帮餐饮商户更了解顾客、商圈，有助于做商业决策。并给顾客带来更好的消费体验（覆盖县市区旗：2800+、用户年均交易笔数：37.2、年度交易用户：6.9亿、业务品类：200+、活跃商户：900万户）。而美团之所以由餐饮C端向餐饮B端延伸：
- C端流量红利不再，存在巨大需求和机会的B端市场有望成为美团的第二增长曲线。随着互联网人口红利的逐渐消失，美团围绕流量的团购、外卖等C端生意逐渐触及到天花板。而围绕B端商家，美团可以在增值服务等方面开发更多的客户，获得更多的增长机会。
- 在C端业务遭遇瓶颈的当下，美团发展B端能反向推动C端，从而加固自身壁垒。B端商家的数字化能力较差，商家的硬件、软件、管理和服务都还没有跟上，各环节效率得不到提升，美团想要打开更大的C端市场就会困难重重。美团To B服务可以全面提升商家的服务品质和效率，而这种对B端的提升最终又能传导到C端，促进C端业务的发展。更何况，美团在C端所拥有的流量、商家、地推优势，都是其发力B端的一大助力。美团拥有庞大的消费者流量及数据资源，可以帮助商家进行精准用户画像；另外美团在餐饮外卖上所积累的商家资源也可以看作其拓展B端市场的最佳客户；而且美团强大的地推能力对其收录商户的数量和质量都能带来一定助力，能更好地促进其B端产品的落地。在C端优势的加持下，美团B端将释放出更大能量，进一步稳固其在生活服务领域的地位。
- 一体化的产品和服务能有效提升协作效率。在过去，餐饮行业业务繁琐，审批流程复杂，线上线下无法成功对接。而一体化的餐饮SaaS系统则涵盖了从点餐、收银、预订、排队到后厨管理、连锁管理及供应链管理等一系列餐饮服务工具，任何一个环节都能在同一系统中实现自动化流转，这不仅可以帮助经营者提升各部门之间的协作效率，还能为消费者提供更高效快捷的体验。
- 一体化的产品和服务能合理降低人力成本。餐企行业特别是连锁品牌，往往具有门店区域分散、人员流动频繁等特点，这也就意味着大部分餐企的人力资源部门需要对员工进行线上招聘、培训、管理和服务等，其中的成本可想而知。而成熟先进的餐饮SaaS已经能从最开始的采购，贯穿到顾客买单、顾客维护、人力管理以及供应链、数据中台等各个环节，能帮助众多餐饮商户缩减人力成本、增加经营收益。
- 一体化的产品和服务能更好地解决供应链痛点。餐饮行业全流程业务紧密相连，数据流、业务流、资金流等任何一方的割裂，都会为餐企开展业务带来重重障碍。比如在疫情中，很多小型或个人餐饮店出现了库存积压困难，进而导致现金流断裂最终不得不选择闭店。而一体化餐饮SaaS系统则可以实现标准化的采购流程，通过数字化分析预测、按需采购，在一定程度上帮助企业开源节流。

##### 挑战

餐饮SaaS面临的挑战：
- 一体化IT解决方案：线上线下一体化，软硬件一体化，店内部署难。线上引流（外卖、团购、预定、排队、买单、评价），线下体验（顾客服务、前厅[点单、支付]收银、餐厅管理[进销存、经营分析、财务管理、会员营销]、后厨管理[KDS、后厨分单、传菜划单]）。
- 业务复杂度高：业态多、业务端多、用户角色多、业务领域多。
- 可用性要求高：餐厅经营重度依赖门店弱网/断网可经营。
- 研发投入大：跨地域、多团队、外包。

![](/img/saas_architecture_1.png "餐饮SaaS面临的挑战")

##### 架构方法

解决方案要求：可信(Trusted，安全、合规、可靠)、易用(Easy，目标明确、自动化、互动性强)、可适应(Adaptable，)。

可信包括：
- 安全：组织安全、会话安全、数据安全，组织安全旨在保护系统免受未经授权的访问。强大的组织安全包括确保只有经过验证的授权用户才能访问系统，并且确保经过验证的用户只能访问必要的功能和数据。会话安全在新窗口中打开链接是一种系统配置方法，旨在防止未经授权的人员通过干扰或劫持会话来访问系统或数据。数据安全是保护数据免遭未经授权的访问、损坏或意外删除的措施。数据安全涉及保护传输中的数据和静态数据。
- 合规：遵守法律、道德标准、无障碍性。
- 可靠：可用性、性能、可扩展性。

易用包括：
- 目标明确：策略、可维护性、可读性。
- 自动化：效率、数据完整性。
- 互动性强：流程化

可适应包括：
- 可弹性：应用的生命周期管理、事件响应、业务连续性规划。
- 可组合：关注点分离、互操作性、可打包性。

机构原则：
- 基础设施即代码：利用领域驱动架构，该原则涉及基础设施的声明性编码、创建不可变的工件以及使用Kubernetes和Service Mesh等标准按需自动化基础设施。
- 零信任安全：实施零信任安全模型，具有全面的防御策略，包括身份管理、身份验证、授权、网络隔离、最小特权安全策略以及传输中和静态数据的加密。
- 托管服务：强调使用多租户和多云服务，该原则增强了跨不同基础设施和环境（如商业、政府和隔离系统）的可移植性。
- 内置弹性：关键任务服务分布在多个可用区，以确保高可用性。数据在可用区域之间复制。服务还带有可用性分层标签，以便管理服务级别目标和弹性规划。
- 完全可观察：将所有服务集成到标准可观察性平台中，以实现高效监控，包括日志收集、指标收集、警报、分布式跟踪以及流量、错误率和资源利用率等服务操作的跟踪。
- 自动化运营：这包括基础设施生命周期的自动化管理和预测性 AIOps（用于运营的 AI），用于维护服务质量、检测和解决服务降级以及故障检测。
- 自动化扩展：该原则注重可扩展性和成本效益，允许跨不同规模的运营灵活性，而不会增加运营风险，抽象与云提供商相关的特定帐户限制。
- 


架构方法：业务架构基于企业架构方法论(TOGAF)，技术架构：采用架构五视图（逻辑架构、开发架构、数据架构、物理架构、运行架构）。架构的关键原则：基于领域驱动设计。

技术架构：
- 架构管控：事前 -架构设计规范、通用能力及组件。事中 -工具检查，定期架构review。事后 -方案设计，架构评审。
- 通用能力及组件：统一的SaaS多租户和管控体系。统一的网关组件(API、流量)。统一的业务集成平台（可插拔平台、开发平台）。店内业务组件(本地服务：localserver)。
- 架构交付物：需求分析（功能&非功能性需求、业务流程）。技术选型与学习最佳实践。架构五视图（逻辑、开发、数据、运行、部署架构）。非功能性设计（安全、稳定性设计）。

##### 研发标准化

研发标准化：
- 设计标准化：设计规范、分层架构、业务平台化。
- 技术栈标准化：模板工程、中间件统一、组件化。
- 服务标准化：微服务、SOA、鉴权组件、脱敏组件、统一灰度、管控组件、限流组件、熔断降级组件、内容审核组件等。
- 测试标准化：CI/CD、sonar、专项测试、自动化测试。
- 运维标准化：灰度、稳定性大盘、故障大盘、故障演练、日志回捞、业务/服务监控、预案管理、值班制度等。

规范统一：项目管理规范、设计规范、代码规范、中间件规范（数据库）、接口规范、环境规范、提测规范、上线规范、报警处理规范等。研发过程度量：研发过程数字化、指标度量系统。
![](/img/saas_architecture.png "餐饮SaaS架构图")

##### 店内经营系统

![](/img/saas_architecture_2.png "餐饮SaaS-店内经营系统的历史演进")

##### 店内经营系统 - 架构

![](/img/saas_architecture_3.png "餐饮SaaS-店内经营系统架构")

##### 店内经营系统 - 交互

![](/img/saas_architecture_4.png "餐饮SaaS-店内经营系统交互")

关键策略：技术问题和业务问题分离，屏蔽技术差异（技术框架）。业务领域化拆分，云店一体化（基础组件建设）。研发和运维流程工具建设，提升研发效率（研发流程支撑）。

###### 技术框架（Local-Server-Framework）

Local-Server-Framework 跨平台开发框架，高高效支支撑服务端同学开发店内业务逻辑，解决复杂 的业务逻辑运行边缘节点的能力，框架涵盖web模块、存储模块、容 器器模块、配置模块、工工具模块和测试模块。框架特点： 
- 支持跨平台（Android、windows、linux）。
- 开发体验好和成本低，基本和Spring开发体验保持一致。
- 低耦合，framework、平台组件、业务模块解耦，业务同学只关注业务代码。
- 性能好，静态编译期处理理，避免动态运行行行时加载，提升启动性能。
- 工具化：打包工具、调试工具、发版工具、运维工具 web 容器。

![](/img/saas_architecture_5.png "餐饮SaaS-店内经营系统-技术框架")

###### 基础组件

![](/img/saas_architecture_6.png "餐饮SaaS-店内经营系统-基础组件")

###### 云店协同组件

店内经营业务： 主数据（SaaS云端）+ 店内业务代码（LocalServer，无状态） = 业务数据（SaaS云端）。主数据即时下发到店内，业务数据及时同步到SaaS云端。
- 数据协同：主数据从云端下发店内：在经营前，商家需要在云端配置好自自己己的菜 品、桌台、支支付方方式等等，需要保证配置数据更更新及时性。业务数据从店内同步云端：经营中产生生的订单、班次上传到云端，做 大大数据分析 
- 业务协同：云店业务实时协同：店内作为餐厅经营的处理理中心心，需要协同处理理云 端顾客扫码点餐业务，保障业务的一致性。

![](/img/saas_architecture_7.png "餐饮SaaS-店内经营系统-云店协同组件")

组件能力：店内业务逻辑开发无无需关注数据协同，配置更更新组件、数据同步组件、业务协同组件。
技术能力：https、长链、外网MQ、外网RPC（上下行，提供丰富的网络通信能力支撑）。

###### 开发模式

![](/img/saas_architecture_8.png "餐饮SaaS-开发模式")

通过框架和工具建设，提升店内业务研发效率：
- 模块化研发：模块化研发，按照领域划分，代码和存储均独 立，模块之间及基于接口交互。
- 开发习惯同Spring框架：IOC、AOP、注解、代码分层、yaml配置。
- 高效打包：动态编译、增量量编译、服务端和Localserver统一打包工具。
- pipeline流水水线：git push流水线、PR流水线、提测流水线。

###### 发布模式

- 发布方式：随客户端发版:商家有感升级，适合大版本发布。静默升级:商家无感升级，适合小版本发布。热更新：hotfix。
- 发布管控：流程规范：需求、编码、测试、集成、发布。质量量卡点：提测卡控、集成卡控、验收卡控、灰度发版。

![](/img/saas_architecture_9.png "餐饮SaaS-发布模式")

###### 灰度模式

问题：
- 店内经营端各个终端和LocalServer如何保证版本兼容和一致？
- 管理端、C端、店内经营端如何保证灰度一致？
- 服务端灰度上线，终端灰度发版，如何能够在保证稳定性情况下，加速灰度节奏，尽快交付商家？ 

解决方方案：
- 沙箱和AB模式两种，服务端全链路路灰度能力。
- 统一灰度池，保证单商户各终端升级版本一致性。
- 支持客户端、前端、后端、数据端等所有端协同灰度。
- 精细化的灰度控制（黑白名单、多维度百分比）。

![](/img/saas_architecture_10.png "餐饮SaaS-灰度模式")

###### 店内监控

业务运行在店内，如何对边缘节点的业务进行监控？App端到端监控（网络成功率、响应时间、运营商等）。业务指标监控（交易易、支支付、接单、打印、消息等）。应用用层监控（异常、请求数、响应时间、存活性）。系统层监控（CPU、内存、磁盘、GC、外设）。

![](/img/saas_architecture_11.png "餐饮SaaS-店内监控")

![](/img/saas_architecture_12.png "美团餐饮-全业务模块SaaS")

###### CRM

客户关系管理(Customer Relationship Management，CRM)，一般而言，只要是服务于销售人员的前端获客、营销人员的品牌管理和售后服务人员的客户服务环节的辅助性软件及工具，都统称为CRM系统。CRM系统服务于面向客户的各业务环节：
- 营销：销售支持、广告投放、产品市场。
- 销售：销售线索、销售协同、销售流程。
- 客服：客服管理、交货实施、二次付费。

![](/img/saas_architecture_13.png "CRM")

营销业务：因为B端营销和C端营销的差异性，从而推出了两套营销方案：
- 面向B端的营销方案（B2B营销）：营销自动化产品(MA)是贯穿全营销过程的基础。了解客户（内部数据结合多渠道第三方数据，建立统一的客户信息认知）、自动化（根据客户触发条件自动发送内容。根据客户行为，指导销售人员进行电销）、触达客户（SEO/SEM、个性化内容全渠道营销投放）、评价营销投入产出（记录客户点击页面的整个行为过程。按渠道分析营销成果）。根据营销端的客户行为反馈进行评级和打分，从而选出最佳潜在客户，提供最优线索，销售人员可以从筛选后的最佳潜在客户群体入手进行拜访，集中精力发现进一步的销售线索和更容易成单的机会，实现销售漏斗更有效转化。
- 面向C端的营销方案（B2C营销）：数据驱动营销，了解用户（帮助客户建立C端用户信息的统一ID和用户画像）、自动化（个性化推荐算法和营销规则）、触达C端用户（个性化内容在社交平台的精准投放。合适的时间将合适的消息发送给合适的人）、评价营销投入产出（电邮、社交媒体、广告、Web和销售平台的跨渠道分析数据和洞察推动投资回报和增长）。例如亚马逊、淘宝等电商平台能够基于用户行为数据提供精准的商品推荐，但传统品牌很难建立起企业与消费者之间的1对1的关系。

总体而言，B2B和B2C营销服务上具有很大区别。其B2B营销服务更注重营销流程的优化和控制，把握住每一次和关键客户沟通交流的机会。而B2C营销更注重用户画像的精准刻画，跨渠道触达和精准投放，以及后续的效果评估。

客服业务：通过智能客服平台对所有客服人员进行统一调度和管理，同时聚合客服人员解决客户难题所需要的所有可得信息，客户的个人信息有助于客服提供个性化反馈。实时聊天软件帮助客服人员通过网络与客户进行全天候实时交流，基于客户问题解决和对等交流的客户社区，提供了客户、客服和企业其他员工更好地互动和更快地解决问题的场所。智能客服助理利用自然语言处理和多模态大模型来处理简单任务。对于复杂任务，智能助理收集并核实客户信息，并将这些信息无缝转交给人工客服。
![](/img/saas_architecture_14.png "客服流程")

电商业务包含B2B和B2C两大模块。电商业务的目标帮助企业（个体商家、KA、品牌方）打造全渠道（手机、社交媒体、Web和门店）智能购买体验，功能包括内容营销、销售、促销活动、客户服务、订单履约、库存管理。统一的消费者体验：
- 线下体验的满足：无论线上还是线下，都可以在线下购买，还可以选择线下或线上退货。智能管理数字渠道和店内的库存、销售、促销活动，一切操作都是通过客户、订单等信息的共享视图来完成。
- 来自店内、手机、社交媒体的全方位数字化消费体验。通过营销、销售和促销等方式吸引顾客并将转化为消费者。打通线上和线下的购物全体验。

![](/img/saas_architecture_15.png "电商C端销售")

传统企业服务软件的两大痛点：
- 因为需要兼顾可定制性和可扩展性，企业级软件产品的用户体验往往没有消费级软件产品的体验好。尤其是2B的开发者很难理解销售、客服、市场等部门的真实工作需求。
- 定制化项目需要大量的二次开发人员，软件研发的成本和需求过大，怎样降低二次开发的难度是关键。

通过低代码的方式来降低开发难度和成本，我在这里是质疑的态度。SaaS订阅模式的成本回收周期很长，这决定了SaaS是长期而精细的慢生意。Salesforce目前走通的商业模式经验表明，SaaS慢生意并不代表回报低，拉长时间轴来看，现金流确定性高且年年延续的生意后期回报丰厚。SaaS营收的主要来源风味两部分：订阅和专业服务。订阅指：客户按月付费、按年结算的方式。“租用”相关服务，既不占有软件，也不占有硬件，只是使用了服务。专业服务是指项目实施、管理以及培训等环节的收费。SaaS订阅模式与软件购买模式的对比：
- SaaS订阅模式：按需租赁，单次付费金额低。对单一客户的收费周期长，迭代升级容易，可以做到一年2-3次更新。按年收费，长期价值高。成本回收周期长，前期经营压力大。面临客户流失的压力，客户数据保存在云端，存在信任问题。
- 购买模式：一次性购买，单词支出高。对单一客户的收费周期短。迭代升级困难，每次升级都要重新进场操作。软件售出后很难在收费。成本回收周期短，经营压力主要在后期。客户数据存在本地，无信任问题。

中小个体商家与KA连锁企业的需求是割裂的，做大企业（KA）的定制化业务，还是面向中小企业（个体商家）的标准化订阅是事关战略战略发展方向的问题。面对中小客户的标准化订阅服务，平均获取客户成本低，无需投入销售人员和后续服务人员成本，但客户生命周期短，订阅账号少，导致客户终身价值和每个用户群平均收入双低，因此总营收的贡献有限。面向大客户的定制化SaaS服务，势必需要前期的高昂的销售成本和后续服务成本、培训成本，并且定制化服务脱离了SaaS模式标准化服务的初心。但好的一面是大客户存续期长，生命周期长叠加转换成本高带来了续费率和长期收入的保障。大客户有固定的销售费用预算，付费习惯、付费能力和付费意愿都全面优于中小企业。判断SaaS模式是否可行及运行健康程度的普适性评价指标：客户终生价值(LTV) / 获客成本(CAC) > 3，客户生命周期价值大于3倍获客成本。根据这个指标能够反映当前阶段SaaS模式的经营状况，该指标决定了SaaS企业的扩张节奏，并提前进行规划。同时该指标也可以指导定价，避免过低的合同单价导致最终亏损。收取获客成本的时间小于12个月。根据这个标准可以判断公司在多长时间内能实现盈利，同时也是对现金流紧张程度的评判。收回获客成本的时间在SaaS模式早期尤为重要，因为创业公司在发展早期通常缺乏资金，获取资金的成本通常也非常高。订阅模式是一项前期投入高而回收成本慢的生意，因此SaaS公司只有准确把握现金流才能渡过持续投入期和业务扩张期。
![](/img/saas_architecture_16.png "SaaS订阅模式的运行指标")

SaaS生态是长期主义思维。B端业务不像C端业务一样具有边际成本几乎为0的特征和赢家通吃的效应，长期来看一家领头B端企业必须依靠于外部生态合作伙伴的互助共赢才能最大程度的扩展业务。长期来看既然市场无法全部吃下，那么尽早地投资潜在力量，以及通过并购吸纳合适的力量可能是最符合长期利益的选择。延迟盈利，持续投资是长期主义。

###### 供应链

供应链就像人体内的循环系统，它负责将养分（原料、零部件、信息、资金）高效、准确地输送到需要的地方（制造、分销、最终消费者），并将代谢产物（退货、回收品、信息反馈）你想输送回去。供应链是一个复杂的、动态的网络系统，它连接了从最初供应商的供应商到最终消费者的消费者之间所有参与创造和交付产品或服务的组织、人员、活动、信息和资源。起点是原始原料；终点是产品或服务被最终消费者使用或消耗；核心目标是在正确的时间、正确的地点，以正确的成本、正确的数量和质量，将正确的产品或服务交付给正确的客户。关键要素是物料流（实物产品）、信息流（订单、预测、状态）、资金流（支付、信用）。

供应链架构：
- 战略层：网络设计-> 决定设施（工厂，仓库，配送中心）的数量、地理位置、规模、功能（制造、组装、存储），这是最基础、影响最深远的决策；外包战略-> 决定哪些活动由企业自己完成，哪些由外部合作伙伴完成；合作伙伴选择与管理-> 识别、评估、选择和管理供应商、合同制造商、物流服务提供商等关键伙伴；产品设计协同-> 在产品设计阶段就考虑供应链的影响（可制造性、可采购性、可运输性）；风险管理战略-> 识别潜在风险（供应中断、需求波动、自然灾害、地缘政治），并制定缓解和应急计划；可持续发展战略-> 将环境（碳排放、资源消耗）、社会（劳工标准、社区影响）、治理(ESG)因素融入供应链决策；技术战略-> 规划支持供应链运营所需的信息系统和数字技术。
- 规划层：需求规划-> 基于历史数据、市场趋势、促销活动等预测未来客户需求。这是驱动整个供应链的源头；销售与运营计划-> 平衡需求预测与供应能力（生产、库存、采购、人力、资金），制定一个统一的、可执行的运营计划；库存规划-> 决定在供应链的哪些节点、持有多少库存（原材料、在制品、成品）以满足服务水平和成本目标；供应/生产规划-> 根据S&OP输出，制定详细的生产排程、原材料采购计划、产能分配计划；分销规划-> 规划如何将成品从生产地或仓库高效地配送到客户或零售点（运输模式、路线、时间）。
- 执行层：采购-> 寻源：寻找和评估潜在供应商；下单：向选定的供应商发出采购订单；收货：接收、检验和入库采购的物料；供应商关系管理：监控供应商绩效（质量、交期、成本），进行持续改进；制造/生产-> 物料准备：根据生产计划将所需物料配送到生产线；生产执行：按照工艺和质量标准进行产品制造或组装；质量控制：在生产过程中和成品后进行质量检验；设备维护：确保生产设备正常运行；物流（仓储与运输）-> 入库： 接收货物、质检、上架存储；存储： 安全、高效地管理库存（库位管理、环境控制）；订单履行： 接收客户订单、拣选、包装、发货准备；出库：装载货物、发货；运输管理：选择承运商、安排运输、跟踪在途货物、管理运费；仓库管理：优化仓库布局、流程和人员效率；退货管理-> 处理客户退回的商品（退货授权、收货、检验、维修/翻新/处置）；管理从客户返回缺陷产品的逆向物流；处理废旧产品的回收再利用或环保处置；
- 支持层：供应链可见性-> 实时或近实时地跟踪物料、产品、订单在供应链中的位置和状态。这是现代供应链高效运作的基础。数据管理与分析-> 收集、整合、分析供应链各环节产生的数据，用于绩效监控、问题诊断、预测优化和决策支持。信息技术系统-> 企业资源规划(ERP)：集成核心业务流程（财务、人力、制造、供应链）的主干系统；供应链管理系统-> 专门用于供应链规划（如需求计划、库存优化、网络设计）和执行（如仓库管理WMS、运输管理TMS、全球贸易管理GTM）的软件套件；供应商关系管理/采购系统-> 管理寻源、采购到付款流程；制造执行系统-> 管理车间级的生产执行；物联网平台-> 连接物理设备（车辆、货物、机器）以获取实时数据；区块链平台-> 提供不可篡改的交易记录，增强透明度和信任（尤其在溯源方面）；绩效管理-> 定义关键绩效指标（如准时交付率、库存周转率、总供应链成本、现金周转周期），持续监控并推动改进；风险管理-> 持续识别、评估、监控和应对供应链运营中的风险；合规管理-> 确保供应链活动符合相关法律法规（贸易法规、产品安全、环保、劳工标准等）；人力管理-> 招聘、培养和留住具备供应链专业知识和技能的人才；
- 贯穿要素：端到端流程-> 强调跨越组织内部部门和外部合作伙伴边界的无缝流程（如订单到现金、采购到付款）；协作-> 供应链各节点（企业内部部门之间、企业与外部合作伙伴之间）的信息共享、联合规划和协同行动至关重要；成本与效率-> 在满足服务水平的前提下，持续优化总供应链成本（采购成本、制造成本、物流成本、库存持有成本等）；敏捷性与韧性-> 供应链需要能够快速响应需求变化（敏捷性）并承受和快速恢复中断（韧性）；创新->持续探索和应用新技术、新流程、新模式以提升供应链竞争力。

供应链架构不是一个静态的流程图，而是一个动态的、相互关联的生态系统。它是一个多层级的框架，从设定方向和构建基础网络的战略层，到平衡供需的规划层，再到具体操作物料、信息和资金流动的执行层，所有这些都需要强大的支持层（技术、数据、人才、流程）来赋能。协作、可见性、客户导向、敏捷性、韧性和成本效率是贯穿整个架构并决定其成功与否的关键原则。


###### 订单系统

订单系统的作用是：管理订单类型、订单状态，收集关于商品、优惠、用户、收货信息、支付信息等一系列的订单实时数据，进行库存更新、订单下发等一系列动作。订单系统的基本模型涉及用户、商品(库存)、订单、付款，订单的基本流程是下订单-> 减库存，这两部必须同时完成，不能下了订单不减库存(超卖)，或者减了库存没有生成订单(少卖)。超卖商家库存不足，消费者下了单买不到东西，体验不好；少卖商家库存积压或者需要反复修改商品信息，体验不好。

订单的多样性：
- 来源：买家、卖家、后台系统自动生成、OpenAPI调用、第三方导入。
- 操作：买家客户订单、买家退货订单、供应商退货订单、买家销售订单、卖家移除订单。

订单字段包含了订单中需要记录的信息，他的作用主要用于沟通其他系统，为下游系统提供信息依据。订单字段包括：
- 订单信息：订单号作为订单识别的标识，一般按照某种特定规则生成，根据订单的增加进行自增，同时在设计订单号的时候考虑订单无序设置（防止竞争者或者第三方来估算订单量）。订单号后续用作订单唯一标识用于对接WMS（仓储管理系统）和TMS（运输管理系统）时的订单识别。订单状态：。
- 用户信息：指卖家的相关信息，包括名称、地址、手机号、收货人、收货地址、收货人联系方式等。O2O还会多一种情况自提点，这样地址则会变为自提点的地址。地址信息在后续会作用在WMS和TMS上用于区分区域和配送安排。
- 商品信息：商品的基本信息（商品属性、商家信息、商品数量等）和库存，金额由于比较特殊，所以把金额独立在商品以外来阐述，不过逻辑上都属于商品信息的范畴。商品信息主要影响库存更新和WMS产生。
- 金额信息：商品单价、支付金额（现金、积分、其它）、应付金额、优惠金额。订单产生的商品信息，这里面除了要记录最终金额，过程金额也需要记录。比如商品分摊的优惠金额、支付金额、应付金额等。在后续的订单结算、退换货、财务等环节都需要使用。
- 时间信息：记录订单每个状态节点的触发时间。如下单时间、支付时间、发货时间、收货时间等。

订单流程：这里面主要是涉及主流电商系统中的通用订单流程，部分细节可以根据自己平台的特殊性进行调整。

订单状态：
- 正向和逆向流程维度：
    - 正向订单：已锁定、已确认、已付款、已发货、已结算、已完成、已取消等。
    - 正向预售订单：预付款已付未确认、已确认未付尾款（变更）。
    - 正向问题单：未确认、未锁定、未发货、部分付款、未付款等。
    - 逆向退单：待结算、未收到货、未入库、质检不通过、部分收货、已取消、客户已收货等。
    - 逆向换单：完成、已结算、客户已收货等。
- 服务对象维度：
    - 顾客/用户：待付款、待发货、待收货、待评价、买家已付款、交易成功/失败、买家已发货、退款成功、交易关闭。
    - ERP等其他交互系统：已锁定、已确认、已分仓、已分配、已出库、已收货、已完成等。
    - 商家：等待买家付款、待付款、待发货、退款中、定金已付、买家已付款、卖家已发货、交易成功/失败。

订单推送：当订单状态发生变化时，需要将对应的订单变化情况告知相关人员以便了解当前订单情况，这就是订单推送的作用。订单推送触发依赖于状态机的变化，涉及到的信息包括：
- 推送对象(用户、物流人员、商家)
- 推送方式(push消息、短信、微信)
- 推送节点（订单状态改变）

订单系统设计的挑战和实践：
- 实现购买流程：1.实现订单的创建、发货、确认等信息闭环。2.支持订单审核（初期可支持人工审核即可）。3.支持用户端显示订单相关信息。4.支持促销金额的计算。
- 提供服务：提供订单分布式服务。支持跨平台交易单生成（即同一个大交易单内既有商家商品又有自营商品或者是多个商家的商品）。支持拆单、合并逻辑（交易单、配送单、支付单等）。提供更丰富的订单推送服务，完善订单状态。
- 支持不同营销手段下的订单类型：平台发展到足够大的规模，提效、稳定变成一个重要的话题。可以提供不同营销场景下的订单，如：团购、预购等。

订单系统的最佳实践：
- 重试和补偿：多个机器重试不能同步, 需要随机跳跃(Jitter)和指数回退 (exponential back-off)。正在重试的服务也可能宕机，需要保存状态 (State)。
- 幂等性：你没收到响应不见得失败了。你响应了不见得别人以为你成功了 。重试必需带上唯一的有意义的ID 。每一个服务的调用都必须是幂等的 。非只读服务必须保存状态。
- 一致性：订单系统有强一致性需求；有时候单点故障并不可怕，常用的，成熟的关系数据库方案也是一个不错的选择。云端分布式无单点故障的系统（TIDB）。
- 工作流(Workflow)：可扩展性、无状态的Worker，分布式部署，分布式存储工作流状态；可靠性：定时器、重试、幂等性、强一致性的状态。可维护性:工作流的描述和执行Activity描述相分离, 支持异步触发。

数据库读写分离：
- 好处：增加冗余、增加了机器的处理能力、对于读操作为主的应用，使用读写分离是最好的场景，因为可以确保写的服务器压力更小，而读又可以接受点时间上的延迟。
- 性能：物理服务器增加，负荷增加。主从只负责各自的写和读，极大程度的缓解X锁和S锁争用。从库可配置myisam引擎，提升查询性能以及节约系统开销。从库同步主库的数据和主库直接写还是有区别的，通过主库发送来的binlog恢复数据，但是，最重要区别在于主库向从库发送binlog是异步的，从库恢复数据也是异步的。读写分离适用与读远大于写的场景，如果只有一台服务器，当select很多时，update和delete会被这些select访问中的数据堵塞，等待select结束，并发性能不高。对于写和读比例相近的应用，应该部署双主相互复制。可以在从库启动是增加一些参数来提高其读的性能，例如--skip-innodb、--skip-bdb、--low-priority-updates以及--delay-key-write=ALL。当然这些设置也是需要根据具体业务需求来定得，不一定能用上。分摊读取。假如我们有1主3从，假设现在1分钟内有10条写入，150条读取。那么，1主3从相当于共计40条写入，而读取总数没变，因此平均下来每台服务器承担了10条写入和50条读取（主库不承担读取操作）。因此，虽然写入没变，但是读取大大分摊了，提高了系统性能。另外，当读取被分摊后，又间接提高了写入的性能。所以，总体性能提高了，说白了就是拿机器和带宽换性能。MySQL复制另外一大功能是增加冗余，提高可用性，当一台数据库服务器宕机后能通过调整另外一台从库来以最快的速度恢复服务，因此不能光看性能，也就是说1主1从也是可以的。
- 数据库分库分表：不管是采用何种分库分表框架或者平台，其核心的思路都是将原本保存在单表中太大的数据进行拆分，将这些数据分散保存到多个数据库的多个表中，避免因为单表数据量太大给数据的访问带来读写性能的问题。所以在分库分表场景下，最重要的一个原则就是被拆分的数据尽可能的平均拆分到后端的数据库中，如果拆分的不均匀，还会产生数据访问热点，同样存在热点数据因为增长过快而又面临数据单表数据量过大的问题。而对于数据以什么样的纬度进行拆分，大家看到很多场景中都是对业务数据的ID(大部分场景此ID是以自增长的方式)进行HASH取模的方式将数据进行平均拆分，这个简单的方式确实在很多场景下都是非常合适的拆分方法，但并不是在所有的场景中这样拆分的方式都是最优的选择。也就是说数据如何拆分并没有所谓的金科玉律，更多的是需要结合业务数据的结构和业务场景来决定。订单数据主要由三张数据库表组成，主订单表对应的就是用户的一个订单，每提交一次都会生成一条主订单表的数据。在有些情况下，用户可能在一个订单中选择不同卖家的商品，而每个卖家又会按照该订单中自己提供的商品计算相关的商品优惠（如满100元减10元）以及按照不同的收货地址设置不同的物流配送，所以会出现子订单的相关概念，即一个主订单会由多个子订单组成，而真正对应到具体每个商品订单信息，则保存在订单详情表中。如果一个电商平台的业务发展健康的话，订单数据是比较容易出现因为单个数据库表中的数据量过大而造成性能的瓶颈，所以需要对他进行数据库的拆分。此时从理论上对订单拆分是可以由两个纬度进行的，一个纬度是通过订单ID（一般为自增长ID）取模的方式，即以订单ID为分库分表键；一个是通过买家用户ID的纬度进行哈希取模，即以买家用户ID为分库分表键。如果是按照订单ID取模的方式，比如按1024取模，则可以保证主订单以及相关子订单，订单详情数据平均落入到后端1024个数据库表中，原则上很好地满足了数据尽可能平均拆分的原则。通过采用买家ID取模的方式，比如也是按照1024取模，技术上则也能保证订单数据拆分到后端的1024个数据库表中，但这里就会出现一个业务场景中带来的问题，就是如果有些卖家是交易量非常大的，那这些卖家的订单数据量（特别是订单详情表的数据量）会比其他卖家要多处不少，也就是会出现数据不平均的现象，最终导致这些卖家的订单数据所在的数据库会相对其他数据库提前进入数据归档（为避免在线交易数据库的数据的增大带来数据库性能的问题，一般将3个月内的订单数据保存在线交易数据库中，超过3个月的订单会归档后端专门的归档数据库）。所以从对『数据尽可能平均拆分』这条原则来看，按照订单ID取模的方式看起来更能保证订单数据的平均拆分，但我们暂时不要这么快下结论，也要根据不同的业务场景和最佳实践角度多思考不同纬度带来的优缺点。

电商平台的需求一直在变化，随之订单系统的架构也会随之变化，架构设计就是一个持续改进的过程，比如：容灾、灾备、分流、流控都需要慢慢雕琢，在架构中没有完美的架构只有平衡的架构，不要追求单点的完美，而是要追求多点的平衡。


