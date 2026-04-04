#|

the purpose of this code is to use LLMs to create an idea tree. 

an idea tree is a tree data structure that has a mission post at the root

a post is a data structure with the fields ptype, name, description, purpose, and achievers.

ptype := mission | stakeholder | goal | barrier | cause | solution | abstraction | analogy | inspiration | question

name and description are strings

purpose is a post

achievers is a list of posts

the key function is build-post. You pass it a post (the purpose) and the type of the achiever you want to 
create for that purpose.

the post-type function returns the ptype of a post

the post-achievers function returns the achievers of a given ptype for a post

the context function returns the chain of posts starting with the one you pass it as an argument going back
to the root of the idea map.

For example, for the following idea tree:

mission a -> stakeholder b -> goal c -> barrier d -> solution f
                                     -> barrier e -> abstraction g -> analogy h -> inspiration i -> solution j -> question k

(purpose b) = a
(post-achievers b) = [ c ]
(post-achievers c) = [d e]
(context d) = [d c b a]

the find-first function searches through the context for a post to find the first post with the given type,
and then returns that e.g.

(find-first '(goal barrier cause) i) = e

(find-first '(answer solution) k) = j

the describe-context function returns the name and description for all of the posts in 
the context for a post in a human-readable way

the propose-achiever function creates a post instance with the given ptype name description and purpose

Your code for creating an idea tree thus will consist of:

1) creating the root for an idea tree (a mission)
2) calling the propose-achiever function repeatedly with existing posts in the idea tree to keep adding to it

how many times you call propose-achiever and what posts you add where are a function of the condition you are testing.

the askllm function sends a prompt to an LLM and returns the response

the parse-llm-response function parses out the name and description from the LLM response

These are the allowable post creation actions, including the ptype of the purpose post, 
the name of the action, and the ptype of the new post:

mission add-stakeholder stakeholder
stakeholder add-goal goal
goal add-barrier barrier
goal add-solution solution
goal add-abstraction abstraction
barrier add-cause cause
barrier add-solution solution
barrier add-abstraction abstraction
cause add-cause cause
cause add-solution solution
cause add-abstraction abstraction
abstraction add-analogy analogy
analogy add-inspiration inspiration
inspiration use-inspiration solution
solution add-improvement solution
solution add-barrier barrier
solution add-question question
question add-answer solution

|#

(defstruct post
 ptype
 purpose
 name
 description
 achievers)

(defun context (post)
  "Returns the ancestors for a post, post followed by parents"
  (when post
    (cons post (context (post-purpose post)))))

(defun find-first (ptypes post)
  "find the first post in the context with one of the given types"
  (unless (listp ptypes) (setq ptypes (list ptypes)))
  (loop for p in (context post)
    when (member (post-ptype p) ptypes)
    do (return p)))

(defun build-post (purpose ptype name description)
  "Creates a new post with the given parameters"
  (let ((new (make-post ptype purpose name description)))
    (pushnew new (post-achievers purpose))))

(defun propose-achiever (&key ptype purpose)
  "Uses the LLM to propose a new post of type ptype as an achiever for the purpose."
  (let* ((existing (loop for p in (post-achievers purpose)
                     when (member (post-ptype p) ptype)
                     collect p))
         (prompt (with-output-to-string (stream)
                   (format stream "~2%Please propose a new ~a" ptype)
                   (unless (eq ptype 'mission) 
                     (format stream " for the ~a: ~a." (post-ptype purpose) (post-name purpose)))
                   (unless (member ptype '(analogy inspiration))
                     (format stream "~2%Do your best to make the ~a suited to the customer's specific context: ~a" ptype (describe-context purpose)))
                   (case ptype
                     (mission (format nil "~2%A mission should describe what deliberation we want to have, including who is the customer, 
                     what decision needs to be made, what we hope to accomplish, the hard constraints on what kinds of solutions are practical (for 
                     example, whether or not it’ll be possible to change laws, what the maximum feasible timeline is, and the magnitude of money that can be spent).
                     Each of those points should appear in their own paragraph prefixed with the topic expressed as a question.")
                              (format stream "~2%Also describe the list of stakeholders who should be considered during the deliberation.")
                              (format stream "~2%Give me your response as a JSON structure as follows:")
                              (format stream "~%```json{")
                              (format stream "~%\"type\": \"~a\"," ptype)
                              (format stream "~%\"description\": <a 2 or 3 sentence description of your proposed mission>," )
                              (format stream "~%\"name\": <a 4 to 6 word title for the mission>," )
                              (format stream "~%}```"))
                     
                     (stakeholder (format stream "~2%A stakeholder is a class of entities whose needs should be considered during the deliberation on ~a. 
                              You should discuss WHY the stakeholder is important for coming up with a successful solution for this deliberation, 
                              for example whether its support is critical for the solution to succeed.
                              Each of those points should appear in their own paragraph prefixed with the topic expressed as a question." (post-name purpose))
                                  (when existing
                                    (format stream "~2%Here are the stakeholders that have been proposed so far:")
                                    (loop for e in existing
                                      do (format stream "~%- ~a ~a" (post-name e) (post-description e))))
                                  (format stream "~2%Pick a stakeholder from the following list that has NOT been already proposed:")
                                  (format stream "~2%Give me your response as a JSON structure as follows:")
                                  (format stream "~%```json{")
                                  (format stream "~%\"type\": \"~a\"," ptype)
                                  (format stream "~%\"description\": <a 2 or 3 sentence description of your proposed stakeholder>," )
                                  (format stream "~%\"name\": <a 2 to 5 word name for the stakeholder>," )
                                  (format stream "~%}```"))
                     
                     (goal (format stream "~2%A goal represents a state (e.g. clean air) that is important for the success of the ~a \"~a\".
                                  You should describe the goal as well as why it is important.
                                  Each of those points should appear in their own paragraph prefixed with the topic expressed as a question." 
                                   (post-ptype purpose) (post-name purpose))
                           (when existing
                             (format stream "~2%The following ~as have already been proposed:" ptype)
                             (loop for e in existing do (format stream "~%- ~a ~a" (post-name e) (post-description e))))
                             (format stream "~2%Make sure your proposed goal includes JUST ONE SIMPLE GOAL.")
                             (format stream "~%For example, DO NOT say \"Secure national energy sovereignty AND affordability\". These are two different goals!")
                             (format stream "~%Pick just one goal e.g. \"Secure national energy sovereignty\".")
                             (format stream "~%DO NOT say \"Insure stable, affordable energy supply\". Again, these are two different goals!")
                             (format stream "~%Pick just one goal e.g. \"Secure stable energy supply\".")
                             (format stream "~2%Give me your response as a JSON structure as follows:")
                             (format stream "~%```json{")
                             (format stream "~%\"type\": \"~a\"," ptype)
                             (format stream "~%\"description\": <a 2 or 3 sentence description of your proposed goal>," )
                             (format stream "~%\"name\": <a 4 to 6 word name for the goal>," )
                             (format stream "~%}```"))
                           
                           (barrier (format stream "~2%A barrier should represent something that undercuts our ability to succeed at the ~a \"~a\".
                             You should describe the barrier as well as why it is an important impediment to the ~a.
                             Each of those points should appear in their own paragraph prefixed with the topic expressed as a question." 
                                            (post-ptype purpose) (post-name purpose) (post-ptype purpose))
                                    (when existing
                                      (format stream "~2%The barrier you propose should be as different as possible from these others:" )
                                      (loop for e in existing
                                        do (format stream "~%- ~a ~a" (post-name e) (post-description e))))
                                    (format stream "~2%Give me your response as a JSON structure as follows:")
                                    (format stream "~%```json{")
                                    (format stream "~%\"type\": \"barrier\",")
                                    (format stream "~%\"description\": <a 2 or 3 sentence description of your proposed barrier>," )
                                    (format stream "~%\"name\": <a 4 to 6 word name for the barrier>" )
                                    (format stream "~%}```"))
                           
                           (cause (format stream "~2%A cause should describe something that can lead to the ~a ~a.
                                    Describe the cause, how it leads to the ~a, and WHY it can have a large impact.
                                    Each of those points should appear in their own paragraph prefixed with the topic expressed as a question." 
                                          (post-ptype purpose) (post-name purpose) (post-ptype purpose))
                                  (when existing
                                    (format stream "~2%The ~a you propose should be as different as possible from these others:" ptype)
                                    (loop for e in existing
                                      do (format stream "~%- ~a ~a" (post-name e) (post-description e))))
                                  (format stream "~2%Give me your response as a JSON structure as follows:")
                                  (format stream "~%```json{")
                                  (format stream "~%\"type\": \"cause\",")
                                  (format stream "~%\"description\": <a 2 or 3 sentence description of your proposed cause>," )
                                  (format stream "~%\"name\": <a 4 to 6 word name for the cause>" )
                                  (format stream "~%}```"))
                           
                           (abstraction (let ((origin (find-first '(goal barrier cause) purpose)))
                                          (format stream "~2%An abstraction is the first step of using analogy to solve a problem. 
                                  The abstraction should represent a generalization of the ~a \"~a\".
                                  Try to generalize the ~a a LOT so I am more likely to get out of the box ideas from the analogy process.
                                  When you generalize it, maintain the deep structure but abstract away domain-specific features e.g.
                                  by replacing nouns and verbs with more general ones (hypernyms) that subsume them."
                                                  (post-ptype origin) (post-name origin) (post-ptype origin))
                                          (when existing
                                            (format stream "~2%The abstraction you propose should be as different as possible from these others:")
                                            (loop for e in existing
                                              do (format stream "~%- ~a ~a" (post-name e) (post-description e))))
                                          (format stream "~2%Give me your response as a JSON structure as follows:")
                                          (format stream "~%```json{")
                                          (format stream "~%\"type\": \"abstraction\",")
                                          (format stream "~%\"description\": <a 2 or 3 sentence description of your proposed abstraction>," )
                                          (format stream "~%\"name\": <a 4 to 6 word name for the abstraction>" )
                                          (format stream "~%}```")))
                           
                           (analogy (let ((origin (find-first '(goal barrier cause) purpose)))
                                      (format stream "~2%An analogy should describe a problem that comes from a different domain than the ~a \"~a\", 
                                          but that instantiates the same structural pattern as the abstraction \"~a\"."
                                              (post-ptype origin) (post-name origin) (post-ptype purpose))
                                      (when existing
                                        (format stream "~2%The analogy you propose should be as different as possible from these others:")
                                        (loop for e in existing
                                          do (format stream "~%- ~a ~a" (post-name e) (post-description e))))
                                      (format stream "~2%Give me your response as a JSON structure as follows:")
                                      (format stream "~%```json{")
                                      (format stream "~%\"type\": \"analogy\",")
                                      (format stream "~%\"description\": <a 2 or 3 sentence description of your proposed analogy>," )
                                      (format stream "~%\"name\": <a 4 to 6 word name for the analogy>" )
                                      (format stream "~%}```")))
                           
                           (improvement (let ((origin (find-first '(goal barrier cause) purpose)))
                                          (format stream "~2%An improvement should describe how the ~a \"~a\" can be made better 
                                      in terms of addressing the ~a ~a. You should discuss HOW it improves upon the ~a, 
                                      and WHY it is a good idea to implement it.
                                      Each of those points should appear in their own paragraph prefixed with the topic expressed as a question."
                                                  (post-ptype purpose) (post-name purpose) (post-ptype origin) (post-name origin) (post-ptype purpose))
                                          (when existing
                                            (format stream "~2%The improvement you propose should be as different as possible from these others:" )
                                            (loop for e in existing
                                              do (format stream "~%- ~a ~a" (post-name e) (post-description e))))
                                          (format stream "~2%Give me your response as a JSON structure as follows:")
                                          (format stream "~%```json{")
                                          (format stream "~%\"type\": \"improvement\",")
                                          (format stream "~%\"description\": <a 2 or 3 sentence description of your proposed improvement>,")
                                          (format stream "~%\"name\": <a 4 to 6 word name for the improvement>" )
                                          (format stream "~%}```")))
                           
                           (inspiration (format stream "~2%Your inspiring idea should represent a possible solution for the ~a \"~a\"." (post-ptype purpose) (post-name purpose))
                                        (when existing
                                          (format stream "~2%The idea you propose should be as different as possible from these others:")
                                          (loop for e in existing
                                            do (format stream "~%- ~a ~a" (post-name e) (post-description e))))
                                        (format stream "~2%Give me your response as a JSON structure as follows:")
                                        (format stream "~%```json{")
                                        (format stream "~%\"type\": \"inspiration\",")
                                        (format stream "~%\"description\": <a 2 or 3 sentence description of your proposed solution>,")
                                        (format stream "~%\"name\": <a 4 to 6 word name for the solution>" )
                                        (format stream "~%}```"))
                           
                           (solution (let ((origin (find-first '(goal barrier cause) purpose)))
                                       (format stream "~2%A solution should address the ~a \"~a\". You should include the following points, each in their own paragraph:
                                        ~2%What is the solution? <2 to 3 sentence description of the solution>
                                        ~2%Why is it a good solution? <2 to 3 sentence description of the advantages of this solution>"
                                               (post-ptype origin) (post-name origin))
                                       (format stream "~2%Just give me the solution, in a succinct way, without re-iterating the problem it solves.")
                                       (when existing
                                         (format stream "~2%The solution you propose should be as different as possible from these others:")
                                         (loop for e in existing
                                           do (format stream "~%- ~a ~a" (post-name e) (post-description e))))
                                       (format stream "~2%Give me your response as a JSON structure as follows:")
                                       (format stream "~%```json{")
                                       (format stream "~%\"type\": \"solution\",")
                                       (format stream "~%\"description\": <description of solution>,")
                                       (format stream "~%\"name\": <a 4 to 6 word name for the solution>" )
                                       (format stream "~%}```")))
                           
                           (question (format stream "~2%Your question should be one whose reply will help make the solution \"~a\" more complete.
                                       A good question could for example identify a possible failure mode with the current solution and ask how to avoid that failure mode.
                                       It could ask what sub-components are needed to make the solution into a reality.
                                       Make sure that the question's name has the structure: how can we X?"
                                             (post-name purpose))
                                     (when existing
                                       (format stream "~2%The ~a you propose should be as different as possible from these others:" ptype)
                                       (loop for e in existing
                                         do (format stream "~%- ~a ~a" (post-name e) (post-description e))))
                                     (format stream "~2%Give me your response as a JSON structure as follows:")
                                     (format stream "~%```json{")
                                     (format stream "~%\"type\": \"question\",")
                                     (format stream "~%\"description\": <a 2 or 3 sentence description of your proposed question>,")
                                     (format stream "~%\"name\": <a 4 to 6 word name for the question>" )
                                     (format stream "~%}```"))
                           
                           (answer (let ((origin (find-first '(answer solution) purpose)))
                                     (format nil "~2%The answer should help make the solution \"~a\" more robust and complete.
                                     You should describe the answer as well as why it is a good one. 
                                     Each of those points should appear in their own paragraph prefixed with the topic expressed as a question." (post-name origin))
                                     (when existing
                                       (format stream "~2%The ~a you propose should be as different as possible from these others:" ptype)
                                       (loop for e in existing
                                         do (format stream "~%- ~a ~a" (post-name e) (post-description e))))
                                     (format stream "~2%Give me your response as a JSON structure as follows:")
                                     (format stream "~%```json{")
                                     (format stream "~%\"type\": \"answer\",")
                                     (format stream "~%\"description\": <a 2 or 3 sentence description of your proposed answer>,")
                                     (format stream "~%\"name\": <a 4 to 6 word name for the answer>" )
                                     (format stream "~%}```"))))))
         (response (askLLM prompt)))
    (multiple-value-bind (name description) (parse-llm-output response)
      (build-post purpose ptype name description))))

(defun describe-context (post)
  "Describes the context for a post in human-readable form"
  (with-output-to-string (stream)
    (loop with chain = (reverse (context post))
      with previous
      while chain
      do (let* ((item (pop chain)) (itemtype (post-ptype item)) (prevtype (post-ptype previous)))
           (when item
             (format stream "~2%")
             (case itemtype
               (mission (format stream "~2%The mission is: ~a." (post-name item)))
               (t (format stream "~2%One ~a for this ~a is: ~a." itemtype prevtype (post-name item))))
             (format stream " ~a" (post-description item))
             (setq previous item))))))

