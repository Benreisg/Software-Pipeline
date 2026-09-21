sys_msg = """
###Your role: 
You are an expert in modeling business processes using BPMN 2.0. Based on a textual description provided by the user, you need to create a valid BPMN process model as an XML string.

### Guidelines:
1. **BPMN XML Format**: The output should be a valid BPMN XML string following the BPMN 2.0 standard.
2. **No Layouting Information**: Exclude any layouting information (do not generate content within `<bpmndi:BPMNDiagram>` tags).
3. **Include All Elements**: Ensure that all tasks, events, pools, lanes, sequence flows and message flows are mentioned in the text description are included accurately.
4. **Add missing information**: You should act as an process ower, which means that you use your expertise to fill in missing information. 
5. **Valid XML**: The XML have to be valid according to the XSD schema. If the model is not valid, you are once asked to correct it.

### Template for an XML
<definitions>  <!-- Root: Declares namespaces, schema, and global elements -->
    <collaboration>  <!-- (Optional) Defines pools, lanes, and message flows -->
        <participant />  
        <messageFlow />
    </collaboration>

    <process>  <!-- Defines the main process with flow elements -->
        <laneSet>  <!-- (Optional) Organizes elements into lanes -->
            <lane />
        </laneSet>

        <!-- Core Flow Elements -->
        <startEvent />
        <task />
        <gateway />
        <sequenceFlow />
        <endEvent />

        <!-- Data & Artifacts -->
        <dataObject />
        <association />
    </process>

    <!-- Reusable Elements -->
    <message />
    <signal />
    <globalTask />
</definitions>


### Output: 
As an output an BPMN in XML format is needed. Use the elements from the BPMN 2.0 notation. Please do only return the XML. Do not explain the modelled process or the xml

### Example Output: 
<?xml version="1.0" encoding="UTF-8"?>
<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL" xmlns:bpmndi="http://www.omg.org/spec/BPMN/20100524/DI" xmlns:di="http://www.omg.org/spec/DD/20100524/DI" xmlns:dc="http://www.omg.org/spec/DD/20100524/DC" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" id="Definitions_1" targetNamespace="http://bpmn.io/schema/bpmn">
  <bpmn:collaboration id="Collaboration_0geryc1">
    <bpmn:participant id="Participant_0ygkg4f" name="Dispatch of goods&#10;Computer Hardware Shop" processRef="Process_1" />
  </bpmn:collaboration>
  <bpmn:process id="Process_1" isExecutable="false">
    <bpmn:laneSet>
      <bpmn:lane id="Lane_1viot5w" name="Logistics">
        <bpmn:flowNodeRef>Task_12j0pib</bpmn:flowNodeRef>
      </bpmn:lane>
      <bpmn:lane id="Lane_1ocseyo" name="Secretary">
        <bpmn:flowNodeRef>Task_0jsoxba</bpmn:flowNodeRef>
        <bpmn:flowNodeRef>Task_0vaxgaa</bpmn:flowNodeRef>
        <bpmn:flowNodeRef>Task_0e6hvnj</bpmn:flowNodeRef>
        <bpmn:flowNodeRef>Task_0s79ile</bpmn:flowNodeRef>
        <bpmn:flowNodeRef>InclusiveGateway_0p2e5vq</bpmn:flowNodeRef>
        <bpmn:flowNodeRef>InclusiveGateway_1dgb4sg</bpmn:flowNodeRef>
        <bpmn:flowNodeRef>StartEvent_1</bpmn:flowNodeRef>
        <bpmn:flowNodeRef>ParallelGateway_02fgrfq</bpmn:flowNodeRef>
        <bpmn:flowNodeRef>ExclusiveGateway_1mpgzhg</bpmn:flowNodeRef>
        <bpmn:flowNodeRef>ExclusiveGateway_1ouv9kf</bpmn:flowNodeRef>
      </bpmn:lane>
      <bpmn:lane id="Lane_1vl2igx" name="Warehouse">
        <bpmn:flowNodeRef>Task_05ftug5</bpmn:flowNodeRef>
        <bpmn:flowNodeRef>Task_0sl26uo</bpmn:flowNodeRef>
        <bpmn:flowNodeRef>ExclusiveGateway_0z5sib0</bpmn:flowNodeRef>
        <bpmn:flowNodeRef>EndEvent_1fx9yp3</bpmn:flowNodeRef>
      </bpmn:lane>
    </bpmn:laneSet>
    <bpmn:sequenceFlow id="SequenceFlow_0iu9po7" name="no" sourceRef="ExclusiveGateway_1mpgzhg" targetRef="InclusiveGateway_0p2e5vq" />
    <bpmn:inclusiveGateway id="InclusiveGateway_0p2e5vq">
      <bpmn:incoming>SequenceFlow_0iu9po7</bpmn:incoming>
      <bpmn:outgoing>SequenceFlow_1j94oja</bpmn:outgoing>
      <bpmn:outgoing>SequenceFlow_1dlbln9</bpmn:outgoing>
    </bpmn:inclusiveGateway>
    <bpmn:task id="Task_12j0pib" name="Insure parcel">
      <bpmn:incoming>SequenceFlow_1j94oja</bpmn:incoming>
      <bpmn:outgoing>SequenceFlow_0kz5g1t</bpmn:outgoing>
    </bpmn:task>
    <bpmn:sequenceFlow id="SequenceFlow_1j94oja" name="If insurance&#10;necessary" sourceRef="InclusiveGateway_0p2e5vq" targetRef="Task_12j0pib" />
    <bpmn:task id="Task_0jsoxba" name="Write package&#10;label">
      <bpmn:incoming>SequenceFlow_1dlbln9</bpmn:incoming>
      <bpmn:outgoing>SequenceFlow_0mp5byl</bpmn:outgoing>
    </bpmn:task>
    <bpmn:sequenceFlow id="SequenceFlow_1dlbln9" name="always" sourceRef="InclusiveGateway_0p2e5vq" targetRef="Task_0jsoxba" />
    <bpmn:sequenceFlow id="SequenceFlow_0mp5byl" sourceRef="Task_0jsoxba" targetRef="InclusiveGateway_1dgb4sg" />
    <bpmn:inclusiveGateway id="InclusiveGateway_1dgb4sg">
      <bpmn:incoming>SequenceFlow_0mp5byl</bpmn:incoming>
      <bpmn:incoming>SequenceFlow_0kz5g1t</bpmn:incoming>
      <bpmn:outgoing>SequenceFlow_0buzwss</bpmn:outgoing>
    </bpmn:inclusiveGateway>
    <bpmn:sequenceFlow id="SequenceFlow_0kz5g1t" sourceRef="Task_12j0pib" targetRef="InclusiveGateway_1dgb4sg" />
    <bpmn:startEvent id="StartEvent_1" name="Ship goods">
      <bpmn:outgoing>SequenceFlow_14a0oky</bpmn:outgoing>
    </bpmn:startEvent>
    <bpmn:parallelGateway id="ParallelGateway_02fgrfq">
      <bpmn:incoming>SequenceFlow_14a0oky</bpmn:incoming>
      <bpmn:outgoing>SequenceFlow_023hzxi</bpmn:outgoing>
      <bpmn:outgoing>SequenceFlow_1ujhfx4</bpmn:outgoing>
    </bpmn:parallelGateway>
    <bpmn:task id="Task_0vaxgaa" name="Clarify shipment method">
      <bpmn:incoming>SequenceFlow_023hzxi</bpmn:incoming>
      <bpmn:outgoing>SequenceFlow_1rss71o</bpmn:outgoing>
    </bpmn:task>
    <bpmn:exclusiveGateway id="ExclusiveGateway_1mpgzhg" name="Special&#10;sandling?">
      <bpmn:incoming>SequenceFlow_1rss71o</bpmn:incoming>
      <bpmn:outgoing>SequenceFlow_0iu9po7</bpmn:outgoing>
      <bpmn:outgoing>SequenceFlow_1xv6wk4</bpmn:outgoing>
    </bpmn:exclusiveGateway>
    <bpmn:sequenceFlow id="SequenceFlow_14a0oky" sourceRef="StartEvent_1" targetRef="ParallelGateway_02fgrfq" />
    <bpmn:sequenceFlow id="SequenceFlow_023hzxi" sourceRef="ParallelGateway_02fgrfq" targetRef="Task_0vaxgaa" />
    <bpmn:sequenceFlow id="SequenceFlow_1rss71o" sourceRef="Task_0vaxgaa" targetRef="ExclusiveGateway_1mpgzhg" />
    <bpmn:task id="Task_0e6hvnj" name="Get 3 offers&#10;from logistic&#10;companies">
      <bpmn:incoming>SequenceFlow_1xv6wk4</bpmn:incoming>
      <bpmn:outgoing>SequenceFlow_1pq8ub3</bpmn:outgoing>
    </bpmn:task>
    <bpmn:sequenceFlow id="SequenceFlow_1xv6wk4" name="yes" sourceRef="ExclusiveGateway_1mpgzhg" targetRef="Task_0e6hvnj" />
    <bpmn:task id="Task_0s79ile" name="Select logistic&#10;company and&#10;place order">
      <bpmn:incoming>SequenceFlow_1pq8ub3</bpmn:incoming>
      <bpmn:outgoing>SequenceFlow_0ajhekx</bpmn:outgoing>
    </bpmn:task>
    <bpmn:sequenceFlow id="SequenceFlow_1pq8ub3" sourceRef="Task_0e6hvnj" targetRef="Task_0s79ile" />
    <bpmn:exclusiveGateway id="ExclusiveGateway_1ouv9kf">
      <bpmn:incoming>SequenceFlow_0ajhekx</bpmn:incoming>
      <bpmn:incoming>SequenceFlow_0buzwss</bpmn:incoming>
      <bpmn:outgoing>SequenceFlow_035vf60</bpmn:outgoing>
    </bpmn:exclusiveGateway>
    <bpmn:sequenceFlow id="SequenceFlow_0ajhekx" sourceRef="Task_0s79ile" targetRef="ExclusiveGateway_1ouv9kf" />
    <bpmn:sequenceFlow id="SequenceFlow_0buzwss" sourceRef="InclusiveGateway_1dgb4sg" targetRef="ExclusiveGateway_1ouv9kf" />
    <bpmn:task id="Task_05ftug5" name="Package goods">
      <bpmn:incoming>SequenceFlow_1ujhfx4</bpmn:incoming>
      <bpmn:outgoing>SequenceFlow_0b2nw5c</bpmn:outgoing>
    </bpmn:task>
    <bpmn:exclusiveGateway id="ExclusiveGateway_0z5sib0">
      <bpmn:incoming>SequenceFlow_035vf60</bpmn:incoming>
      <bpmn:incoming>SequenceFlow_0b2nw5c</bpmn:incoming>
      <bpmn:outgoing>SequenceFlow_06kfaev</bpmn:outgoing>
    </bpmn:exclusiveGateway>
    <bpmn:sequenceFlow id="SequenceFlow_035vf60" sourceRef="ExclusiveGateway_1ouv9kf" targetRef="ExclusiveGateway_0z5sib0" />
    <bpmn:sequenceFlow id="SequenceFlow_0b2nw5c" sourceRef="Task_05ftug5" targetRef="ExclusiveGateway_0z5sib0" />
    <bpmn:task id="Task_0sl26uo" name="Prepare for&#10;picking up&#10;goods">
      <bpmn:incoming>SequenceFlow_06kfaev</bpmn:incoming>
      <bpmn:outgoing>SequenceFlow_0v64x8b</bpmn:outgoing>
    </bpmn:task>
    <bpmn:sequenceFlow id="SequenceFlow_06kfaev" sourceRef="ExclusiveGateway_0z5sib0" targetRef="Task_0sl26uo" />
    <bpmn:endEvent id="EndEvent_1fx9yp3" name="Shipment&#10;prepared">
      <bpmn:incoming>SequenceFlow_0v64x8b</bpmn:incoming>
    </bpmn:endEvent>
    <bpmn:sequenceFlow id="SequenceFlow_0v64x8b" sourceRef="Task_0sl26uo" targetRef="EndEvent_1fx9yp3" />
    <bpmn:sequenceFlow id="SequenceFlow_1ujhfx4" sourceRef="ParallelGateway_02fgrfq" targetRef="Task_05ftug5" />
  </bpmn:process>
</bpmn:definitions>

"""

inavlid_xml = """
It seems that the XML format is invalid. 
Ensure that:
- the id of the elements are unique
- elements are defined at the right point of the xml 
- each sequence and message flow has a sourceRef and targetRef attribute
Make sure to correct all the mistakes given in the following list and just return the corrected XML, without any further explanation:

"""

modeling_prompt = """
Create a BPMN model in XML format for the following textual description of a process. Make sure to follow the BPMN 2.0 modeling guidelines.  

### Process description:
"""