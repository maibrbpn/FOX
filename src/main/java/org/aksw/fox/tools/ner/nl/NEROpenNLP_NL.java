package org.aksw.fox.tools.ner.nl;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

import opennlp.tools.namefind.NameFinderME;
import opennlp.tools.namefind.TokenNameFinderModel;
import opennlp.tools.util.Span;

import org.aksw.fox.data.Entity;
import org.aksw.fox.data.EntityClassMap;
import org.aksw.fox.tools.ner.AbstractNER;
import org.aksw.fox.tools.ner.en.NEROpenNLP;
import org.aksw.fox.utils.FoxCfg;
import org.aksw.fox.utils.FoxConst;
import org.aksw.fox.utils.FoxTextUtil;
import org.apache.log4j.PropertyConfigurator;

public class NEROpenNLP_NL extends AbstractNER {

    private final String[]  modelPath = { "data/openNLP/nl-ner-person.bin", "data/openNLP/nl-ner-location.bin", "data/openNLP/nl-ner-organization.bin" };

    private final TokenNameFinderModel[] tokenNameFinderModels = new TokenNameFinderModel[modelPath.length];

    /**
     * 
     */
    public NEROpenNLP_NL() {
        InputStream[] modelIn = new InputStream[3];
        for (int i = 0; i < tokenNameFinderModels.length; i++) {
            try {

                modelIn[i] = new FileInputStream(modelPath[i]);
                if (modelIn[i] != null)
                    tokenNameFinderModels[i] = new TokenNameFinderModel(modelIn[i]);

            } catch (IOException e) {
                LOG.error("\n", e);
            } finally {

                try {
                    if (modelIn[i] != null)
                        modelIn[i].close();
                } catch (IOException e) {
                    LOG.error("\n", e);
                }
            }
        }
    }

    // TODO: do parallel for each model
    @Override
    public List<Entity> retrieve(String input) {
        LOG.info("retrieve ...");

        List<Entity> list = new ArrayList<>();
        String[] sentences = FoxTextUtil.getSentences(input);

        for (int i = 0; i < tokenNameFinderModels.length; i++) {
            if (tokenNameFinderModels[i] != null) {
                NameFinderME nameFinder = new NameFinderME(tokenNameFinderModels[i]);
                for (String sentence : sentences) {
                    String[] tokens = FoxTextUtil.getSentenceToken(sentence);

                    if (tokens.length > 0 && tokens[tokens.length - 1].trim().isEmpty())
                        tokens[tokens.length - 1] = ".";

                    Span[] nameSpans = nameFinder.find(tokens);
                    double[] probs = nameFinder.probs(nameSpans);
                    for (int k = 0; k < nameSpans.length; k++) {
                        Span span = nameSpans[k];

                        String word = "";
                        for (int j = 0; j < span.getEnd() - span.getStart(); j++)
                            word += tokens[span.getStart() + j] + " ";
                        word = word.trim();
                        float p = Entity.DEFAULT_RELEVANCE;
                        if (FoxCfg.get("openNLPDefaultRelevance") != null && !Boolean.valueOf(FoxCfg.get("openNLPDefaultRelevance")))
                            p = Double.valueOf(probs[k]).floatValue();
                        String cl = EntityClassMap.openNLP(span.getType());
                        if (cl != EntityClassMap.getNullCategory())
                            list.add(getEntity(word, cl, p, getToolName()));
                    }
                }
                nameFinder.clearAdaptiveData();
            }
        }
        // TRACE
        if (LOG.isTraceEnabled()) {
            LOG.trace(list);
        } // TRACE
        return list;
    }

    public static void main(String[] a) {
        PropertyConfigurator.configure(FoxCfg.LOG_FILE);

        for (Entity e : new NEROpenNLP_NL().retrieve(FoxConst.NER_NL_EXAMPLE_2))
        {
            NEROpenNLP_NL.LOG.info(e);
        }
        System.out.println("Fertig");
       
    }
   
}
